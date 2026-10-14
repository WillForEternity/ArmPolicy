"""Train the main looped-MLP policy on GPU and save it in policy/ format (policy.pt + normalizer.json),
so `python -m policy.export <run> --loops 4 6 8` works unchanged.

Same recipe as policy/train.py (masked L1 on normalized chunks, AdamW + warmup + cosine, clip 1, EMA 0.999,
K sampled per batch, last 4 loops supervised, state/box noise), with the GPU loader from arch_sweep.py.
Real teleop episodes (same npz layout as sim_bridge output) can be co-trained with --real, upweighted by
--real-frac (fraction of each batch drawn from real frames).

    python train_main.py --data sim_episodes_v1.npz sim_episodes_v2.npz --out runs/main
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from arch_sweep import GPUChunks, evaluate, load, masked_l1, sample_loops, split
from model import LoopedMLPPolicy  # policy/model.py


class Wrap(torch.nn.Module):
    """arch_sweep's (x -> flat chunk) API around LoopedMLPPolicy, for evaluate()."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, loops=8, supervise_last=1):
        return self.m(x[:, :6], x[:, 6:], loops, supervise_last).flatten(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--real", nargs="*", default=[])
    p.add_argument("--real-frac", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--chunk", type=int, default=30)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--loop-mean", type=float, default=6.0)
    p.add_argument("--loop-max", type=int, default=12)
    p.add_argument("--supervise-last", type=int, default=4)
    p.add_argument("--state-noise", type=float, default=0.05)
    p.add_argument("--box-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sim_tr, sim_va = split(load(a.data))
    real_tr, real_va = split(load(a.real)) if a.real else ([], [])
    fit_eps = sim_tr + real_tr
    stats_np = {k: (np.concatenate([e[k] for e in fit_eps]).mean(0),
                    np.maximum(np.concatenate([e[k] for e in fit_eps]).std(0), 1e-2)) for k in ("state", "action", "box")}
    stats = {k: tuple(torch.tensor(v, device=dev, dtype=torch.float32) for v in s) for k, s in stats_np.items()}
    sim = GPUChunks(sim_tr, stats, a.chunk, dev)
    real = GPUChunks(real_tr, stats, a.chunk, dev) if real_tr else None
    vals = {"sim": GPUChunks(sim_va, stats, a.chunk, dev)}
    if real_va:
        vals["real"] = GPUChunks(real_va, stats, a.chunk, dev)

    model = LoopedMLPPolicy(box_dim=4, chunk=a.chunk, d=a.d).to(dev)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    n_params = sum(q.numel() for q in model.parameters())
    print(f"looped_mlp d={a.d}: {n_params/1e3:.1f}k params; sim {len(sim_tr)}/{len(sim_va)} eps, real {len(real_tr)}/{len(real_va)} eps")

    t0 = time.time()
    n_real = int(a.batch * a.real_frac) if real else 0
    for step in range(1, a.steps + 1):
        xs, acts, pads = [], [], []
        for src, n in ((sim, a.batch - n_real), (real, n_real)):
            if n:
                x, act, pad = src.batch(torch.randint(0, src.n, (n,), device=dev), a.state_noise, a.box_noise)
                xs.append(x); acts.append(act); pads.append(pad)
        x, act, pad = torch.cat(xs), torch.cat(acts), torch.cat(pads)
        k = sample_loops(a.loop_mean, k_max=a.loop_max)
        pred = model(x[:, :6], x[:, 6:], k, a.supervise_last)
        loss = masked_l1(pred, act, pad)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update_parameters(model)
        if step % 2000 == 0 or step == a.steps:
            ema.eval()
            errs = {n: evaluate(Wrap(ema.module), v, stats["action"][1], 8, a.chunk)[0] for n, v in vals.items()}
            print(f"step {step:6d} loss {loss.item():.4f} " + " ".join(f"val_{n} {e:.3f}" for n, e in errs.items())
                  + f" ({time.time()-t0:.0f}s)", flush=True)

    ema.eval(); m = ema.module
    curve = {k: evaluate(Wrap(m), vals["sim"], stats["action"][1], k, a.chunk) for k in range(1, 2 * a.loop_max + 1)}
    k8_joints = curve[8][1]
    curve = {k: v[0] for k, v in curve.items()}
    cfg = {"box_dim": 4, "chunk": a.chunk, "d": a.d}
    torch.save({"model": "looped_mlp", "cfg": cfg, "state_dict": m.state_dict()}, out / "policy.pt")
    (out / "normalizer.json").write_text(json.dumps({k: {"mean": s[0].tolist(), "std": s[1].tolist()} for k, s in stats_np.items()}))
    (out / "metrics.json").write_text(json.dumps({"args": vars(a), "params": n_params, "val_err_vs_loops": curve,
                                                  "per_joint_k8": k8_joints}, indent=1))
    print("val err vs K:", {k: round(v, 3) for k, v in curve.items()})
    print("per-joint err @K=8 (pan, lift, elbow, wflex, roll, grip):", [round(v, 2) for v in k8_joints])
    print("TRAIN_DONE", out)


if __name__ == "__main__":
    main()
