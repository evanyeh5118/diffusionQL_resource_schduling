import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .Helpers import DiffusionSchedule, EMATarget
from .Actors import DiffusionPolicy
from .Critics import TD3_IQL


class DQL_Q_iql():
    """Holds two Q-networks and their EMA targets, with update logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        N_diffusion_steps: int = 30,
        approximate_action: bool = True,
        gamma: float = 0.99,
        abs_action_max: float = 1.0,
        lr: float = 2e-4,
        decay_lr: bool = False,
        weight_decay: float = 0.1,
        iql_tau: float = 0.7,
        weight_entropy_loss: float = 1.0,
        weight_q_loss: float = 1.0,
        ema_tau: float = 0.005,
        ema_period: int = 10,
        ema_begin_update: int = 1000,
        layer_norm: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.abs_action_max = abs_action_max
        self.sched = DiffusionSchedule(N_diffusion_steps).to(self.device)
        self.critic = TD3_IQL(
             state_dim=state_dim, action_dim=action_dim, 
            gamma=gamma, lr=lr, decay_lr=decay_lr, weight_decay=weight_decay, iql_tau=iql_tau, 
            ema_tau=ema_tau, ema_period=ema_period, ema_begin_update=ema_begin_update, 
            device=device, layer_norm=layer_norm).to(self.device)
        self.actor = DiffusionPolicy(state_dim, action_dim, self.sched, layer_norm=layer_norm).to(self.device)
        self.actor_target = EMATarget(self.actor, ema_tau).to(self.device)
        self.optimizer_actor = torch.optim.Adam(
            list(self.actor.parameters()), lr=lr, weight_decay=weight_decay
        )
        self.ema_period = ema_period
        self.ema_begin_update = ema_begin_update
        self.ema_steps = 0
        self.decay_lr = decay_lr
        if self.decay_lr == True:
            self.scheduler_actor = CosineAnnealingLR(self.optimizer_actor, T_max=1000, eta_min=0.)
        self.approximate_action = approximate_action
        self.weight_entropy_loss = weight_entropy_loss
        self.weight_q_loss = weight_q_loss
    
    def update(self, batch_off, batch_on=None):
        if batch_on is not None:
            s,a = batch_on[0], batch_on[1]
            s_off, a_off = batch_off[0], batch_off[1]
        else:
            s,a = batch_off[0], batch_off[1]

        if batch_on is not None:
            loss_Q, _ = self.critic.update(batch_on)
            Ld = self.actor.diffusion_loss(s_off, a_off)
        else:
            loss_Q, _ = self.critic.update(batch_off)
            Ld = self.actor.diffusion_loss(s, a)
        
        # 3) Q-improvement loss
        a0_hat, a1_hat = self.actor.approximate_action(s, a) 
        if np.random.rand() < 0.5:
            q_loss = self.critic.q1(s, a0_hat) / self.critic.q2(s, a0_hat).abs().mean().detach()
        else:
            q_loss = self.critic.q2(s, a0_hat) / self.critic.q1(s, a0_hat).abs().mean().detach()
        Lq = -q_loss.mean()       
        # 4) Entropy loss
        L_entropy = self.actor.entropy_loss(s, a0_hat, a1_hat)
        Le = -L_entropy.mean() / L_entropy.abs().mean().detach()                       # scalar
        # 5) Combined loss
        loss_pi = Ld + self.weight_q_loss * Lq + self.weight_entropy_loss * Le
        # 6) Backprop
        self.optimizer_actor.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0, norm_type=2)
        self.optimizer_actor.step()
        # 7) EMA update
        if self.decay_lr == True:
            self.scheduler_actor.step()
        if self.ema_steps % self.ema_period == 0 and self.ema_steps > self.ema_begin_update:
            self.actor_target.soft_update()
        self.ema_steps += 1
 
        return Ld.item(), Lq.item(), Le.item(), loss_Q

    def train(self, replay_buffer, iterations, batch_size, tqdm_pos=0):
        metrics = {"Ld": [], "Lq": [], "Le": [], "loss_Q": []}
        for i in tqdm(range(iterations), position=tqdm_pos, leave=False):
            batch = replay_buffer.sample(batch_size)
            Ld, Lq, Le, loss_Q = self.update(batch)
            metrics["Ld"].append(Ld)
            metrics["Lq"].append(Lq)
            metrics["Le"].append(Le)
            metrics["loss_Q"].append(loss_Q)
        return metrics
    
    def train_split(self, offline_buffer, online_buffer, iterations, batch_size, tqdm_pos=0):
        metrics = {"Ld": [], "Lq": [], "Le": [], "loss_Q": []}
        for i in tqdm(range(iterations), position=tqdm_pos, leave=False):
            batch_offline = offline_buffer.sample(batch_size)
            batch_online = online_buffer.sample(batch_size)
            Ld, Lq, Le, loss_Q = self.update(batch_offline, batch_online)
            metrics["Ld"].append(Ld)
            metrics["Lq"].append(Lq)
            metrics["Le"].append(Le)
            metrics["loss_Q"].append(loss_Q)
        return metrics
    
    @torch.no_grad()
    def sample(self, 
            s: torch.Tensor, 
            N: int = 10, 
            eta: float = 0.1,
            sample_method: str = "greedy",
            add_noise: bool = False,
        ) -> torch.Tensor:
        B = s.size(0)
        s_rep = s.unsqueeze(1).expand(B, N, s.size(-1)).reshape(-1, s.size(-1))
        a_cand = self.actor.sample_DDIM(s_rep, eta=eta)
        q_cand = self.critic.q_min(s_rep, a_cand).view(B, N)     
        a_cand = a_cand.view(B, N, -1)  
        if sample_method == "greedy":
            best_idx = torch.argmax(q_cand, dim=1)                     # (B,)
            a_best = a_cand[torch.arange(B), best_idx]  # (B, action_dim)
        elif sample_method == "EAS":
            mean, std = q_cand.mean(1, keepdim=True), q_cand.std(1, keepdim=True)
            logits = (q_cand - mean) / (std + 1e-6)
            probs  = torch.softmax(logits, dim=1)             # (B,N) 
            idx = torch.multinomial(probs, num_samples=1)     # (B,1)
            a_best = a_cand[torch.arange(B, device=s.device), idx.squeeze(-1)]
        else:
            raise ValueError(f"Invalid sample method: {sample_method}")
        if add_noise:
            a_best = a_best + torch.randn_like(a_best) * 0.1
        a_best = a_best.clamp(-self.abs_action_max, self.abs_action_max)
        return a_best

    @torch.no_grad()
    def sample_bc(self, s: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
        return self.actor.sample_DDIM(s, eta=eta).clamp(-self.abs_action_max, self.abs_action_max)

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:   
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

