"""Architecture sweep: MLP vs looped MLP vs graph-encoder + looped MLP (vs the looped transformer).

Same task, recipe and metric as policy/train.py so numbers are comparable:
(state 6, box 4) -> 30-step action chunk, masked L1 on normalized actions, AdamW + warmup + cosine, grad clip 1,
EMA 0.999, state/box noise 0.05, episode-level 15% validation split (seed 0); val error = mean |joint error| in
LeRobot units (deg; gripper 0-100). Data loading is vectorized on the GPU (same sampling semantics as
policy.data.ChunkDataset) so a run takes a minute or two.

    python arch_sweep.py --model looped_mlp --d 128 --data sim_episodes_v1.npz sim_episodes_v2.npz --out runs/x
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# ---------------------------------------------------------------- models


class MLP(nn.Module):
    """policy.model.MLPPolicy: 2 hidden layers."""

    def __init__(self, in_dim, out_dim, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x, loops=1, supervise_last=1):
        return self.net(x)[None]


class LoopedMLP(nn.Module):
    """One weight-shared residual MLP block iterated K times with input re-injection:
    h <- norm(h + ff(norm_in(h + x0))). Same looping recipe as policy.model.LoopedPolicy (Yang et al. 2311.12424,
    Geiping et al. 2502.05171): no loop-index embedding, K sampled per batch, last loops supervised."""

    def __init__(self, in_dim, out_dim, d=128, ff_mult=4, dropout=0.1, encoder=None):
        super().__init__()
        self.encoder = encoder or nn.Linear(in_dim, d)
        self.norm_in = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, out_dim)

    def step(self, h, x0):
        return self.norm(h + self.ff(self.norm_in(h + x0)))

    def forward(self, x, loops=6, supervise_last=1):
        x0 = self.encoder(x)
        h = torch.zeros_like(x0)
        window = min(supervise_last, loops)
        with torch.set_grad_enabled(torch.is_grad_enabled() and loops == window):
            for _ in range(loops - window):
                h = self.step(h, x0)
        outs = []
        for _ in range(window):
            h = self.step(h, x0)
            outs.append(self.head(h))
        return torch.stack(outs)


class StackedMLP(LoopedMLP):
    """Control for looping: the same residual block structure but `layers` separate (unshared) blocks."""

    def __init__(self, in_dim, out_dim, d=128, ff_mult=4, dropout=0.1, layers=8):
        super().__init__(in_dim, out_dim, d, ff_mult, dropout)
        mk = lambda: nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))  # noqa: E731
        self.ffs = nn.ModuleList(mk() for _ in range(layers))
        self.norms_in = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        del self.ff

    def forward(self, x, loops=None, supervise_last=1):
        x0 = self.encoder(x)
        h = torch.zeros_like(x0)
        for ff, n_in in zip(self.ffs, self.norms_in):
            h = self.norm(h + ff(n_in(h + x0)))
        return self.head(h)[None]


class GraphEncoder(nn.Module):
    """Tiny GNN over the fixed robot graph: 6 joint nodes (kinematic chain) + 1 box node linked to every joint.
    Node features: joint value * learned direction + joint id; box = linear(box features). R rounds of
    weight-shared message passing (mean over neighbours, MLP update), then concat-readout -> d."""

    def __init__(self, d=128, dn=32, rounds=2, n_joints=6, box_dim=4):
        super().__init__()
        self.nj, self.rounds = n_joints, rounds
        self.val = nn.Parameter(torch.randn(n_joints, dn) * 0.1)
        self.jid = nn.Parameter(torch.randn(n_joints, dn) * 0.1)
        self.box = nn.Linear(box_dim, dn)
        n = n_joints + 1
        A = torch.zeros(n, n)
        for i in range(n_joints - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        A[:n_joints, n_joints] = A[n_joints, :n_joints] = 1
        self.register_buffer("A", A / A.sum(1, keepdim=True))
        self.msg = nn.Sequential(nn.Linear(2 * dn, 2 * dn), nn.GELU(), nn.Linear(2 * dn, dn))
        self.norm = nn.LayerNorm(dn)
        self.out = nn.Linear(n * dn, d)

    def forward(self, x):
        s, b = x[:, : self.nj], x[:, self.nj:]
        h = torch.cat([s[..., None] * self.val + self.jid, self.box(b)[:, None]], 1)  # (B, 7, dn)
        for _ in range(self.rounds):
            m = self.A @ h
            h = self.norm(h + self.msg(torch.cat([h, m], -1)))
        return self.out(h.flatten(1))


class LoopedTF(nn.Module):
    """Reference: the policy session's looped transformer (policy/model.py), unchanged, behind this script's API."""

    def __init__(self, d, chunk):
        super().__init__()
        from model import LoopedPolicy  # policy/model.py copied next to this script
        self.m = LoopedPolicy(box_dim=4, chunk=chunk, d=d)

    def forward(self, x, loops=6, supervise_last=1):
        y = self.m(x[:, :6], x[:, 6:], loops, supervise_last)
        return y.flatten(2)


def build(name, in_dim, out_dim, d):
    if name == "looped_tf":
        return LoopedTF(d, out_dim // 6)
    if name == "mlp":
        return MLP(in_dim, out_dim, hidden=d)
    if name == "stacked_mlp":
        return StackedMLP(in_dim, out_dim, d=d, layers=8)
    if name == "looped_mlp":
        return LoopedMLP(in_dim, out_dim, d=d)
    if name == "graph_looped_mlp":
        return LoopedMLP(in_dim, out_dim, d=d, encoder=GraphEncoder(d=d, dn=max(16, d // 4)))
    raise ValueError(name)


def macs(model, name, loops):
    """Multiply-accumulates per inference (Linear layers + graph matmuls), loops counted."""
    lin = lambda m: m.in_features * m.out_features  # noqa: E731
    if name == "looped_tf":
        d, n = model.m.box.out_features, 6 + 1 + model.m.chunk
        return n * d * 2 + loops * (n * 12 * d * d + 2 * n * n * d) + model.m.chunk * d * 6
    if name in ("mlp", "stacked_mlp"):
        return sum(lin(m) for m in model.modules() if isinstance(m, nn.Linear))
    step = sum(lin(m) for m in model.ff if isinstance(m, nn.Linear))
    enc = sum(lin(m) for m in model.encoder.modules() if isinstance(m, nn.Linear))
    if isinstance(model.encoder, GraphEncoder):
        g = model.encoder
        per_node_msg = sum(lin(m) for m in g.msg if isinstance(m, nn.Linear))
        enc = 7 * lin(g.box) // 7 + g.rounds * (7 * per_node_msg + 49 * g.val.shape[1]) + lin(g.out)
    return enc + loops * step + lin(model.head)


# ---------------------------------------------------------------- data (mirrors policy.data)


def load(paths):
    eps = []
    for p in paths:
        d = np.load(p)
        n = len({k.split("_")[0] for k in d.files if k.startswith("ep")})
        eps += [{k: d[f"ep{i}_{k}"].astype(np.float32) for k in ("state", "action", "box")} for i in range(n)]
    return eps


def split(eps, val_frac=0.15, seed=0):
    order = np.random.default_rng(seed).permutation(len(eps))
    n_val = max(1, round(val_frac * len(eps)))
    return [eps[i] for i in order[n_val:]], [eps[i] for i in order[:n_val]]


class GPUChunks:
    """All frames of all episodes on the GPU; batch = random frames -> (x_t, action[t:t+H], pad)."""

    def __init__(self, eps, stats, chunk, dev):
        cat = lambda k: torch.tensor(np.concatenate([e[k] for e in eps]), device=dev)  # noqa: E731
        nz = lambda k, x: (x - stats[k][0]) / stats[k][1]  # noqa: E731
        self.state, self.box, self.action = nz("state", cat("state")), nz("box", cat("box")), nz("action", cat("action"))
        lens = torch.tensor([len(e["state"]) for e in eps], device=dev)
        ends = torch.cumsum(lens, 0)
        self.end = torch.repeat_interleave(ends, lens) - 1  # last frame index of each frame's episode
        self.n, self.chunk, self.dev = len(self.state), chunk, dev

    def batch(self, idx, state_noise=0.0, box_noise=0.0):
        steps = idx[:, None] + torch.arange(self.chunk, device=self.dev)
        pad = steps > self.end[idx, None]
        steps = torch.minimum(steps, self.end[idx, None])
        s = self.state[idx] + state_noise * torch.randn_like(self.state[idx])
        b = self.box[idx] + box_noise * torch.randn_like(self.box[idx])
        return torch.cat([s, b], -1), self.action[steps], pad


def sample_loops(mean=6.0, sigma=0.5, k_max=12, p_short=0.1):
    if np.random.rand() < p_short:
        return int(np.random.randint(1, 3))
    tau = np.random.normal(math.log(mean) - sigma**2 / 2, sigma)
    return int(min(k_max, np.random.poisson(math.exp(tau)) + 1))


def masked_l1(pred, target, pad):
    keep = (~pad)[None, :, :, None].float()
    return ((pred - target[None]).abs() * keep).sum() / (keep.sum() * pred.shape[-1] * pred.shape[0])


@torch.no_grad()
def evaluate(model, val, a_std, loops, chunk):
    tot = cnt = 0.0
    per_joint = torch.zeros(6, device=val.dev)
    for i in range(0, val.n, 8192):
        idx = torch.arange(i, min(i + 8192, val.n), device=val.dev)
        x, a, pad = val.batch(idx)
        pred = model(x, loops)[-1].view(len(idx), chunk, 6)
        keep = (~pad)[..., None].float()
        err = (pred - a).abs() * a_std * keep
        tot += err.sum().item(); cnt += keep.sum().item() * 6
        per_joint += err.sum((0, 1))
    return tot / cnt, (per_joint / (cnt / 6)).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mlp", "stacked_mlp", "looped_mlp", "graph_looped_mlp", "looped_tf"])
    p.add_argument("--d", type=int, default=128, help="hidden width (MLP) / state width (looped)")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=30)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--loop-mean", type=float, default=6.0)
    p.add_argument("--loop-max", type=int, default=12)
    p.add_argument("--supervise-last", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    eps = load(a.data)
    tr, va = split(eps)
    stats = {}
    for k in ("state", "action", "box"):
        x = np.concatenate([e[k] for e in tr])
        stats[k] = (torch.tensor(x.mean(0), device=dev), torch.tensor(np.maximum(x.std(0), 1e-2), device=dev))
    train, val = GPUChunks(tr, stats, a.chunk, dev), GPUChunks(va, stats, a.chunk, dev)
    looped = a.model not in ("mlp", "stacked_mlp")
    model = build(a.model, 10, a.chunk * 6, a.d).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    print(f"{a.model} d={a.d}: {n_params/1e3:.1f}k params, {len(tr)} train / {len(va)} val episodes, {train.n} frames")

    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, train.n, (a.batch,), device=dev)
        x, act, pad = train.batch(idx, 0.05, 0.05)
        k = sample_loops(a.loop_mean, k_max=a.loop_max) if looped else 1
        pred = model(x, k, a.supervise_last).view(-1, a.batch, a.chunk, 6)
        loss = masked_l1(pred, act, pad)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update_parameters(model)
        if step % 2000 == 0 or step == a.steps:
            ema.eval()
            err, _ = evaluate(ema.module, val, stats["action"][1], round(a.loop_mean), a.chunk)
            print(f"step {step:6d} loss {loss.item():.4f} val_err {err:.3f} ({time.time()-t0:.0f}s)", flush=True)

    ema.eval(); m = ema.module
    curve, joints = {}, {}
    for k in (range(1, 2 * a.loop_max + 1) if looped else [1]):
        curve[k], joints[k] = evaluate(m, val, stats["action"][1], k, a.chunk)
    best_k = min(curve, key=curve.get)
    res = {"model": a.model, "d": a.d, "params": n_params, "best_k": best_k, "best_err": curve[best_k],
           "macs_at_best_k": macs(m, a.model, best_k), "err_vs_k": curve, "per_joint_at_best_k": joints[best_k],
           "train_s": time.time() - t0, "args": vars(a)}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    torch.save({"model": a.model, "d": a.d, "state_dict": m.state_dict(),
                "stats": {k: [v.tolist() for v in s] for k, s in stats.items()}}, out / "policy.pt")
    print("RESULT", json.dumps({k: res[k] for k in ("model", "d", "params", "best_k", "best_err", "macs_at_best_k")}))
    print("err_vs_k", {k: round(v, 3) for k, v in curve.items()})


if __name__ == "__main__":
    main()
