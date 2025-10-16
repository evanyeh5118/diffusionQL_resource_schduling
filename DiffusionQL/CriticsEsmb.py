import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Helpers import SinusoidalPosEmb

from typing import Tuple
import math

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 3, "shape should be [num_models, batch_size, in_features]"
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ensemble_size={self.ensemble_size}'

class EnsembleCritic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            num_critics: int = 100,
            layernorm: bool = False,
            edac_init: bool = True
    ):
        super().__init__()
        self.ensemble = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, 1, num_critics)
        )
        if edac_init:
            for layer in self.ensemble[::3]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.ensemble[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(self.ensemble[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        out = self.ensemble(state_action).squeeze(-1)
        return out.permute(1,0)

    def q_min(self, state, action):
        # we dont have two to avoid over-estimation
        out = self.forward(state, action)
        return out


class EnsembleDoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            num_critics: int = 100,
            layernorm: bool = False,
            edac_init: bool = True
    ):
        super().__init__()
        self.ensemble1 = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, 1, num_critics)
        )
        self.ensemble2 = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.Mish(),
            VectorizedLinear(hidden_dim, 1, num_critics)
        )

        if edac_init:
            for layer in self.ensemble1[::3]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.ensemble1[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(self.ensemble1[-1].bias, -3e-3, 3e-3)

            for layer in self.ensemble2[::3]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.ensemble2[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(self.ensemble2[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        out1 = self.ensemble1(state_action).squeeze(-1)
        out2 = self.ensemble2(state_action).squeeze(-1)
        return out1.permute(1,0), out2.permute(1,0) # [batch_size, num_critic]

    def q_min(self, state, action):
        out1, out2 = self.forward(state, action)
        return torch.min(out1, out2)