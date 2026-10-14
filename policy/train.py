"""Train a policy on episodes and report held-out error per loop count.

    python -m policy.train --data fake --model looped --out runs/fake_looped
    python -m policy.train --data data/box_pick --model stacked --out runs/stacked

Recipe: masked L1 on normalized action chunks (ACT), AdamW + warmup + cosine, grad clip 1,
EMA 0.999 (TRM 2510.04871). The looped model samples K per batch and supervises the last
`--supervise-last` loops, so every K is a usable policy at run time.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .data import ChunkDataset, Normalizer, fake_episodes, split_episodes
from .model import LoopedMLPPolicy, LoopedPolicy, MLPPolicy, StackedPolicy, sample_loops

MODELS = {"looped": LoopedPolicy, "stacked": StackedPolicy, "mlp": MLPPolicy, "looped_mlp": LoopedMLPPolicy}
LOOPED = ("looped", "looped_mlp")


def load_episodes(data: str) -> list[dict]:
    if data == "fake":
        return fake_episodes(50)
    from .dataset_io import load_lerobot_episodes

    return load_lerobot_episodes(Path(data))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """pred (L, B, H, A) vs target (B, H, A); padded steps excluded."""
    keep = (~pad)[None, :, :, None].float()
    return ((pred - target[None]).abs() * keep).sum() / (keep.sum() * pred.shape[-1] * pred.shape[0])


@torch.no_grad()
def evaluate(model, loader, normalizer: Normalizer, device, loops: int) -> float:
    """Mean absolute joint error in LeRobot units over held-out chunks."""
    std = torch.tensor(normalizer.stats["action"]["std"], device=device)
    total, count = 0.0, 0.0
    for state, box, action, pad in loader:
        state, box, action, pad = (x.to(device) for x in (state, box, action, pad))
        pred = model(state, box, loops)[-1]
        keep = (~pad)[..., None].float()
        total += ((pred - action).abs() * std * keep).sum().item()
        count += keep.sum().item() * action.shape[-1]
    return total / count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="fake")
    p.add_argument("--model", choices=MODELS, default="looped")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=30)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--steps", type=int, default=8000)
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
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args.data)
    train_eps, val_eps = split_episodes(episodes)
    normalizer = Normalizer.fit(train_eps)
    train_ds = ChunkDataset(train_eps, normalizer, args.chunk, args.state_noise, args.box_noise)
    val_ds = ChunkDataset(val_eps, normalizer, args.chunk)
    train_dl = torch.utils.data.DataLoader(train_ds, args.batch, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, 1024)
    box_dim = train_eps[0]["box"].shape[1]
    print(f"{len(train_eps)} train / {len(val_eps)} val episodes, {len(train_ds)} train frames, device={device}")

    cfg = {"box_dim": box_dim, "chunk": args.chunk}
    if args.model != "mlp":
        cfg["d"] = args.d
    model = MODELS[args.model](**cfg).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))
    )

    step, t0 = 0, time.time()
    while step < args.steps:
        for state, box, action, pad in train_dl:
            state, box, action, pad = (x.to(device) for x in (state, box, action, pad))
            loops = sample_loops(rng, args.loop_mean, k_max=args.loop_max) if args.model in LOOPED else 1
            pred = model(state, box, loops, args.supervise_last)
            loss = masked_l1(pred, action, pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            step += 1
            if step % 500 == 0 or step == args.steps:
                ema.eval()
                err = evaluate(ema.module, val_dl, normalizer, device, args.loop_mean.__round__())
                print(f"step {step:6d}  loss {loss.item():.4f}  val_err {err:.3f}  ({time.time() - t0:.0f}s)")
            if step >= args.steps:
                break

    ema.eval()
    final = ema.module
    curve = {}
    for k in range(1, 2 * args.loop_max + 1) if args.model in LOOPED else [1]:
        curve[k] = evaluate(final, val_dl, normalizer, device, k)
    print("val joint error vs loops:", {k: round(v, 3) for k, v in curve.items()})

    torch.save({"model": args.model, "cfg": cfg, "state_dict": final.state_dict()}, out / "policy.pt")
    (out / "normalizer.json").write_text(json.dumps(normalizer.to_json()))
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "val_err_vs_loops": curve}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
