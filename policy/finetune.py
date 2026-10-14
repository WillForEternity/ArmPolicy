"""Fine-tune a trained policy (e.g. the sim-only looped MLP) on real teleop demos.

Starts from the run's weights and keeps its normalizer, so the exported graph and the board see
the same input/output scaling as the sim policy. Batches mix real and sim chunks (--real-frac) so
the sim coverage of the workspace isn't forgotten. Two real episodes are held out to report the
real-world chunk error of the sim policy before fine-tuning vs. after.

    python -m policy.finetune --init runs/main_provisional --real datasets/real_pick_ep0 \
        --sim sim_data/sim_episodes_v1_PROVISIONAL.npz sim_data/sim_episodes_v2_PROVISIONAL.npz \
        --out runs/finetune_real
then: python -m policy.export runs/finetune_real --loops 4 6 8
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .data import Normalizer
from .model import LoopedMLPPolicy, sample_loops


def trim_static_start(e: dict, deg=3.0, keep=3) -> dict:
    """Drop the frames before the operator starts moving (keeping `keep`). Demos open with ~1 s of the
    arm sitting at rest, which teaches 'wait' at the rest pose; with re-planning every few steps the
    policy then re-plans the wait forever and never leaves rest."""
    d = np.abs(e["action"][:, :5] - e["action"][0, :5]).max(1)
    start = max(0, int(np.argmax(d > deg)) - keep) if (d > deg).any() else 0
    return {k: v[start:] for k, v in e.items()}


def load_sim(paths: list[str]) -> list[dict]:
    eps = []
    for p in paths:
        z = np.load(p)
        n = len({k.split("_")[0] for k in z.files if k.startswith("ep")})
        eps += [{k: z[f"ep{i}_{k}"].astype(np.float32) for k in ("state", "action", "box")} for i in range(n)]
    return eps


class Chunks:
    """All (state_t, box_t, action[t:t+H], pad) of a set of episodes, normalized, as tensors."""

    def __init__(self, episodes: list[dict], norm: Normalizer, chunk: int):
        s, b, a, pad = [], [], [], []
        for e in episodes:
            T = len(e["action"])
            idx = np.arange(T)[:, None] + np.arange(chunk)[None]
            pad.append(idx >= T)
            a.append(norm.norm("action", e["action"][np.minimum(idx, T - 1)]))
            s.append(norm.norm("state", e["state"]))
            b.append(norm.norm("box", e["box"][:, :4]))
        self.s, self.b, self.a = (torch.tensor(np.concatenate(x), dtype=torch.float32) for x in (s, b, a))
        self.pad = torch.tensor(np.concatenate(pad))
        self.n = len(self.s)

    def batch(self, n: int, state_noise: float, box_noise: float):
        i = torch.randint(0, self.n, (n,))
        s = self.s[i] + state_noise * torch.randn(n, self.s.shape[1])
        b = self.b[i] + box_noise * torch.randn(n, self.b.shape[1])
        return s, b, self.a[i], self.pad[i]


def masked_l1(pred, target, pad):
    keep = (~pad)[None, :, :, None].float()
    return ((pred - target[None]).abs() * keep).sum() / (keep.sum() * pred.shape[-1] * pred.shape[0])


@torch.no_grad()
def chunk_error(model, data: Chunks, norm: Normalizer, loops=8) -> tuple[float, list[float]]:
    """Mean absolute error of the predicted action chunk in real units (degrees / gripper 0-100)."""
    model.eval()
    pred = model(data.s, data.b, loops, 1)[-1]
    err = (pred - data.a).abs() * torch.tensor(norm.stats["action"]["std"])
    keep = (~data.pad)[..., None].float()
    per_joint = ((err * keep).sum((0, 1)) / keep.sum()).tolist()
    return float(np.mean(per_joint)), per_joint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="run dir with policy.pt + normalizer.json")
    p.add_argument("--real", required=True, help="LeRobot v3 dataset of real teleop demos")
    p.add_argument("--sim", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--real-frac", type=float, default=0.75)
    p.add_argument("--sim-eps", type=int, default=1500, help="random subset of sim episodes to mix in")
    p.add_argument("--holdout", type=int, default=2, help="real episodes held out for validation")
    p.add_argument("--keep-static-start", action="store_true", help="don't trim the idle frames before motion")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--loop-mean", type=float, default=6.0)
    p.add_argument("--loop-max", type=int, default=12)
    p.add_argument("--supervise-last", type=int, default=4)
    p.add_argument("--state-noise", type=float, default=0.05)
    p.add_argument("--box-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    init, out = Path(a.init), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(init / "policy.pt", map_location="cpu", weights_only=False)
    assert ckpt["model"] == "looped_mlp", f"only looped_mlp runs are supported, got {ckpt['model']}"
    cfg = ckpt["cfg"]
    norm = Normalizer(json.loads((init / "normalizer.json").read_text()))
    model = LoopedMLPPolicy(**cfg)
    model.load_state_dict(ckpt["state_dict"])

    from .dataset_io import load_lerobot_episodes

    real = load_lerobot_episodes(Path(a.real))
    if not a.keep_static_start:
        before_n = sum(len(e["action"]) for e in real)
        real = [trim_static_start(e) for e in real]
        print(f"trimmed idle starts: {before_n} -> {sum(len(e['action']) for e in real)} real frames", flush=True)
    order = rng.permutation(len(real))
    real_va = [real[i] for i in order[: a.holdout]]
    real_tr = [real[i] for i in order[a.holdout:]]
    sim = load_sim(a.sim)
    if len(sim) > a.sim_eps:
        sim = [sim[i] for i in rng.choice(len(sim), a.sim_eps, replace=False)]
    print(f"real: {len(real_tr)} train / {len(real_va)} held out ({sum(len(e['action']) for e in real_tr)} frames); "
          f"sim: {len(sim)} episodes", flush=True)

    tr_real = Chunks(real_tr, norm, cfg["chunk"])
    tr_sim = Chunks(sim, norm, cfg["chunk"]) if sim else None
    va_real = Chunks(real_va, norm, cfg["chunk"]) if real_va else None

    before = chunk_error(model, va_real or tr_real, norm)
    print(f"sim policy on held-out real demos: {before[0]:.2f} mean abs error per joint (deg / gripper %)", flush=True)

    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.995))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    n_real = a.batch if tr_sim is None else int(a.batch * a.real_frac)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        parts = [tr_real.batch(n_real, a.state_noise, a.box_noise)]
        if tr_sim is not None and a.batch - n_real:
            parts.append(tr_sim.batch(a.batch - n_real, a.state_noise, a.box_noise))
        s, b, act, pad = (torch.cat(x) for x in zip(*parts))
        k = sample_loops(gen, a.loop_mean, k_max=a.loop_max)
        loss = masked_l1(model(s, b, k, a.supervise_last), act, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update_parameters(model)
        if step % 500 == 0 or step == a.steps:
            err = chunk_error(ema.module, va_real or tr_real, norm)[0]
            print(f"step {step:5d} loss {loss.item():.4f} held-out real err {err:.2f} ({time.time() - t0:.0f}s)", flush=True)

    m = ema.module
    after = chunk_error(m, va_real or tr_real, norm)
    sim_err = chunk_error(m, tr_sim, norm)[0] if tr_sim is not None else None
    torch.save({"model": "looped_mlp", "cfg": cfg, "state_dict": m.state_dict()}, out / "policy.pt")
    shutil.copy(init / "normalizer.json", out / "normalizer.json")
    metrics = {"args": vars(a), "init": str(init), "real_train_eps": len(real_tr), "real_heldout_eps": len(real_va),
               "heldout_real_err_before": before[0], "heldout_real_err_after": after[0],
               "heldout_real_per_joint_after": after[1], "sim_err_after": sim_err}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"held-out real error: sim policy {before[0]:.2f} -> fine-tuned {after[0]:.2f}; "
          f"per joint (pan, lift, elbow, wflex, roll, grip) {[round(x, 1) for x in after[1]]}")
    print("FINETUNE_DONE", out)


if __name__ == "__main__":
    main()
