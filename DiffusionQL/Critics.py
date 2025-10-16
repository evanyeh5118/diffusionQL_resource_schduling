
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .Helpers import EMATarget, mlp

class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (32, 32, 32),
        layer_norm: bool = True
    ):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1, hidden=hidden_sizes, p_dropout=0.05, layer_norm=layer_norm)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)
    
class VNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: tuple[int, ...] = (32, 32, 32),
        layer_norm: bool = False
    ):
        super().__init__()  
        self.net = mlp(state_dim, 1, hidden=hidden_sizes, p_dropout=0.05, layer_norm=layer_norm)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)

class TD3_IQL(nn.Module):
    """Holds two Q-networks and their EMA targets, with update logic."""

    def __init__(self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 3e-4,
        decay_lr: bool = False,
        weight_decay: float = 0.0001,
        iql_tau: float = 0.5,
        ema_tau: float = 0.005,
        ema_period: int = 10,
        ema_begin_update: int = 1000,
        layer_norm: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.gamma = gamma
        self.q1 = QNetwork(state_dim, action_dim, layer_norm=layer_norm).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, layer_norm=layer_norm).to(self.device)
        self.q1_target = EMATarget(self.q1, ema_tau).to(self.device)
        self.q2_target = EMATarget(self.q2, ema_tau).to(self.device)
        self.optimizer_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr, weight_decay=weight_decay
        )
        self.v_net = VNetwork(state_dim, layer_norm=layer_norm).to(self.device)
        self.v_net_target = EMATarget(self.v_net, ema_tau).to(self.device)  
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.iql_tau = iql_tau
        self.decay_lr = decay_lr    
        if decay_lr == True:
            self.scheduler_q = CosineAnnealingLR(self.optimizer_q, T_max=1000, eta_min=0.)
            self.scheduler_v = CosineAnnealingLR(self.optimizer_v, T_max=1000, eta_min=0.)  
        self.ema_period = ema_period
        self.ema_begin_update = ema_begin_update
        self.ema_steps = 0
        
    def _expectile_loss(self, diff, tau):
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * (diff ** 2)).mean()

    def update(self, batch):
        s, a, r, s_next = batch
        # ============================================
        # Update V network via expectile regression 
        # ============================================
        with torch.no_grad():
            q1, q2 = self.q1_target(s, a), self.q2_target(s, a)
            q_min = torch.min(q1, q2)
        loss_v = self._expectile_loss(
            q_min.detach()-self.v_net(s), 
            self.iql_tau
        )
        self.optimizer_v.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 1.0, norm_type=2)
        self.optimizer_v.step()
        # ============================================
        # Update Q networks
        # ============================================
        with torch.no_grad():
            #v_next = self.v_net_target(s_next)
            v_next = self.v_net(s_next)
            q_target = (r + self.gamma * v_next).detach()
        q1_pred, q2_pred = self.q1(s, a), self.q2(s, a)
        loss_q = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0, norm_type=2)
        self.optimizer_q.step()

        # ============================================
        if self.decay_lr == True:
            self.scheduler_q.step()
            self.scheduler_v.step()
        if self.ema_steps % self.ema_period == 0 and self.ema_steps > self.ema_begin_update:
            #self.v_net_target.soft_update()
            self.q1_target.soft_update()
            self.q2_target.soft_update()
        self.ema_steps += 1
        return loss_q.item(), loss_v.item()
    
    def q_min(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.q1_target(s, a), self.q2_target(s, a)
        return torch.min(q1, q2)

class TD3(nn.Module):
    """Holds two Q-networks and their EMA targets, with update logic."""

    def __init__(self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 3e-4,
        decay_lr: bool = False,
        weight_decay: float = 0.1,
        ema_tau: float = 0.005,
        ema_period: int = 10,
        ema_begin_update: int = 1000,
        device: str = "cpu",
    ):
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = EMATarget(self.q1, ema_tau).to(self.device)
        self.q2_target = EMATarget(self.q2, ema_tau).to(self.device)
        self.optimizer_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr, weight_decay=weight_decay
        )
        self.decay_lr = decay_lr    
        if decay_lr == True:
            self.scheduler_q = CosineAnnealingLR(self.optimizer_q, T_max=1000, eta_min=0.)
        self.ema_period = ema_period
        self.ema_begin_update = ema_begin_update
        self.ema_steps = 0
        
    def update(self, batch, a_next):
        s, a, r, s_next = batch
        # ============================================
        # Update Q networks
        # ============================================
        with torch.no_grad():
            noise  = (0.05*torch.randn_like(a_next)).clamp(-0.5, 0.5)
            a_next = (a_next + noise).clamp(-1.0, 1.0)
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_target = r + self.gamma * torch.min(q1_next, q2_next)
        q1_pred, q2_pred = self.q1(s, a), self.q2(s, a)
        loss_q = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
        self.optimizer_q.step()
        if self.decay_lr == True:
            self.scheduler_q.step()
        # ============================================
        # EMA update
        # ============================================
        if self.ema_steps % self.ema_period == 0 and self.ema_steps > self.ema_begin_update:
            self.q1_target.soft_update()
            self.q2_target.soft_update()
        self.ema_steps += 1
        return loss_q.item()

    
