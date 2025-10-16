import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Sequence
import math
import copy

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def timestep_embedding(t: torch.Tensor, dim: int, *, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal time-step embedding (same as DDPM)."""
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freq[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, dim % 2))  # zero-pad if odd


def mlp(inp: int, out: int, hidden: Sequence[int] = (128, 128, 128), p_dropout: float = 0.2, layer_norm: bool = True):
    mods: list[nn.Module] = []
    prev = inp
    for h in hidden:
        mods.append(nn.Linear(prev, h))
        if layer_norm:
            mods.append(nn.LayerNorm(h))
        mods.append(nn.Mish())
        mods.append(nn.Dropout(p_dropout))
        prev = h
    mods.append(nn.Linear(prev, out))
    return nn.Sequential(*mods)

@dataclass
class DiffusionSchedule:
    N: int = 30                 # number of noise steps
    beta_min: float = 0.1       # only used for VP schedule
    beta_max: float = 1.0       # only used for VP schedule
    schedule_type: str = 'vp'   # 'vp' for variance-preserving or 'cosine'

    def __post_init__(self):
        if self.schedule_type == 'vp':
            # Variance-preserving (VP) schedule
            i = torch.arange(1, self.N + 1, dtype=torch.float32)
            self.beta = 1.0 - torch.exp(
                -self.beta_min / self.N
                - 0.5 * (self.beta_max - self.beta_min) * (2 * i - 1) / (self.N ** 2)
            )
            self.alpha = 1.0 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        else:
            raise ValueError(f"Unknown schedule_type '{self.schedule_type}', choose 'vp' or 'cosine'.")

    def to(self, device: torch.device):
        # move tensors to target device
        for attr in ('beta', 'alpha', 'alpha_bar'):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    
class EMATarget(nn.Module):
    """Wraps a network to maintain a frozen EMA-smoothed copy."""

    def __init__(self, source: nn.Module, tau: float = 0.005):
        """
        Args:
            source: the network to track
            tau: smoothing coefficient (0 < tau ≤ 1)
        """
        super().__init__()
        self.source = source
        self.tau = tau
        # Deep-copy registers `target` as a sub-module
        self.target = copy.deepcopy(source)
        self.freeze()

    def freeze(self) -> None:
        """Disable gradients for the target network."""
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def soft_update(self) -> None:
        """Perform in-place polyak averaging of target towards source."""
        for tp, sp in zip(self.target.parameters(), self.source.parameters()):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * sp.data)
            
    @torch.no_grad()
    def forward(self, *args, **kwargs):  # type: ignore[override]
        # Delegate forward to the target network
        return self.target(*args, **kwargs)
    
