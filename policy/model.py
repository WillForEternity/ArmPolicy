"""Looped-transformer policy: one weight-shared transformer block applied K times.

Tokens: one per joint, one for the box pose, and `chunk` learned action queries. The embedded input
is re-injected (added) before every loop, s <- norm(block(s + e)) (Yang et al. 2311.12424), with an
extra norm on the state after each loop (Geiping et al. 2502.05171). There is no loop-index
embedding, so the block is a pure iterated map and K can run past the trained range. K is sampled
per batch during training and chosen at run time (accuracy vs. latency, as in RD-VLA 2602.07845).
The action queries' states are decoded into a chunk of future joint targets, as in ACT.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    """Pre-norm transformer block written with plain ops so it exports cleanly to ONNX."""

    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = F.softmax(q @ k.transpose(-2, -1) * (d // self.heads) ** -0.5, dim=-1)
        x = x + self.drop(self.proj((self.drop(att) @ v).transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.norm2(x)))


class LoopedPolicy(nn.Module):
    def __init__(
        self,
        state_dim=6,
        box_dim=6,
        act_dim=6,
        chunk=30,
        d=128,
        heads=4,
        ff_mult=4,
        dropout=0.1,
    ):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim
        self.joint_val = nn.Parameter(torch.randn(state_dim, d) * 0.02)  # each joint token = value * own direction
        self.joint_id = nn.Parameter(torch.randn(state_dim, d) * 0.02)
        self.box = nn.Linear(box_dim, d)
        self.queries = nn.Parameter(torch.randn(chunk, d) * 0.02)
        self.block = Block(d, heads, ff_mult, dropout)
        self.norm = nn.LayerNorm(d)  # keeps the recurrent state's scale fixed across loops
        self.head = nn.Linear(d, act_dim)

    def embed(self, state: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        joints = state[..., None] * self.joint_val + self.joint_id
        return torch.cat([joints, self.box(box)[:, None], self.queries.expand(b, -1, -1)], dim=1)

    def step(self, h: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return self.norm(self.block(h + x0))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h[:, -self.chunk :])

    def forward(self, state: torch.Tensor, box: torch.Tensor, loops: int, supervise_last: int = 1) -> torch.Tensor:
        """state (B, 6), box (B, 6) normalized -> actions (supervise_last, B, chunk, act_dim) normalized.

        Returns the decoded chunk after each of the last `supervise_last` loops; the loops before
        that window run without gradients (truncated backprop, Geiping et al. / RD-VLA).
        """
        x0 = self.embed(state, box)
        h = torch.zeros_like(x0)
        window = min(supervise_last, loops)
        with torch.set_grad_enabled(torch.is_grad_enabled() and loops == window):
            for _ in range(loops - window):
                h = self.step(h, x0)
        outs = []
        for _ in range(window):
            h = self.step(h, x0)
            outs.append(self.decode(h))
        return torch.stack(outs)


def sample_loops(rng: torch.Generator, mean=6.0, sigma=0.5, k_max=12, p_short=0.1) -> int:
    """Log-normal Poisson loop count (Geiping et al. 2502.05171), scaled down, with a floor that
    occasionally picks 1-2 loops so the whole accuracy-vs-K curve stays trained."""
    if torch.rand((), generator=rng) < p_short:
        return int(torch.randint(1, 3, (), generator=rng))
    tau = torch.normal(math.log(mean) - sigma**2 / 2, sigma, (), generator=rng)
    return int(min(k_max, torch.poisson(tau.exp(), generator=rng) + 1))


class StackedPolicy(LoopedPolicy):
    """Baseline: same tokens and head, but K separate blocks (no weight sharing)."""

    def __init__(self, layers=6, **kw):
        super().__init__(**kw)
        d = self.box.out_features
        heads, ff_mult, dropout = self.block.heads, self.block.ff[0].out_features // d, self.block.drop.p
        self.blocks = nn.ModuleList(Block(d, heads, ff_mult, dropout) for _ in range(layers))
        del self.block

    def forward(self, state, box, loops=None, supervise_last=1):
        x0 = self.embed(state, box)
        h = torch.zeros_like(x0)
        for blk in self.blocks:
            h = self.norm(blk(h + x0))
        return self.decode(h)[None]


class MLPPolicy(nn.Module):
    """Baseline: plain MLP from (state, box) to the flattened action chunk."""

    def __init__(self, state_dim=6, box_dim=6, act_dim=6, chunk=30, hidden=512, dropout=0.1):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + box_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, chunk * act_dim),
        )

    def forward(self, state, box, loops=None, supervise_last=1):
        return self.net(torch.cat([state, box], -1)).view(1, -1, self.chunk, self.act_dim)


class LoopedMLPPolicy(nn.Module):
    """Looped MLP: one weight-shared residual MLP block iterated K times with input re-injection,
    h <- norm(h + ff(norm_in(h + x0))), same looping recipe as LoopedPolicy (no loop-index embedding,
    K sampled per batch, last loops supervised). Beat a 361k-param MLP with 46k params in the
    arch sweep (experiments/arch_sweep.py) and, unlike the looped transformer, stays flat past the
    trained K, so K is a safe accuracy/latency dial on the board.
    """

    def __init__(self, state_dim=6, box_dim=4, act_dim=6, chunk=30, d=128, ff_mult=4, dropout=0.1):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim
        self.encoder = nn.Linear(state_dim + box_dim, d)
        self.norm_in = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, chunk * act_dim)

    def step(self, h, x0):
        return self.norm(h + self.ff(self.norm_in(h + x0)))

    def forward(self, state, box, loops=8, supervise_last=1):
        """state (B, 6), box (B, box_dim) normalized -> (supervise_last, B, chunk, act_dim) normalized."""
        x0 = self.encoder(torch.cat([state, box], -1))
        h = torch.zeros_like(x0)
        window = min(supervise_last, loops)
        with torch.set_grad_enabled(torch.is_grad_enabled() and loops == window):
            for _ in range(loops - window):
                h = self.step(h, x0)
        outs = []
        for _ in range(window):
            h = self.step(h, x0)
            outs.append(self.head(h).view(-1, self.chunk, self.act_dim))
        return torch.stack(outs)
