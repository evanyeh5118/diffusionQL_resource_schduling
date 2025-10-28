import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .Helpers import DiffusionSchedule, EMATarget
from .Actors import DiffusionPolicy
from .CriticsEsmb import EnsembleDoubleCritic

class DQL_Q_esmb():
    """Holds two Q-networks and their EMA targets, with update logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        N_diffusion_steps: int = 30,
        schedule_type: str = "vp",
        approximate_action: bool = True,
        gamma: float = 0.99,
        abs_action_max: float = 1.0,
        lr: float = 2e-4,
        decay_lr: bool = False,
        weight_decay: float = 0.0001,
        num_critics: int = 8,
        lcb_coef: float = 0.5,
        q_sample_eta: float = 0.5,
        max_q_backup: bool = False,
        weight_entropy_loss: float = 0.005,
        weight_q_loss: float = 1.0,
        ema_tau: float = 0.005,
        ema_period: int = 10,
        ema_begin_update: int = 1000,
        layer_norm: bool = False,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.abs_action_max = abs_action_max
        
        self.critic = EnsembleDoubleCritic(
            state_dim, action_dim, hidden_dim=128, num_critics=num_critics, layernorm=False).to(device)
        self.critic_target = EMATarget(self.critic, ema_tau).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.sched = DiffusionSchedule(N_diffusion_steps, schedule_type=schedule_type).to(self.device)
        self.actor = DiffusionPolicy(state_dim, action_dim, self.sched, hidden_dim=128, layer_norm=layer_norm).to(self.device)
        self.actor_target = EMATarget(self.actor, ema_tau).to(self.device)
        self.optimizer_actor = torch.optim.Adam(
            list(self.actor.parameters()), lr=lr
        )
        self.ema_period = ema_period
        self.ema_begin_update = ema_begin_update
        self.ema_steps = 0
        self.decay_lr = decay_lr
        if self.decay_lr == True:
            self.scheduler_lr_actor = CosineAnnealingLR(self.optimizer_actor, T_max=1000, eta_min=0.)
            self.scheduler_lr_critic = CosineAnnealingLR(self.critic_optimizer, T_max=1000, eta_min=0.)
        self.approximate_action = approximate_action
        self.weight_bc_loss = 1.0
        self.weight_entropy_loss = weight_entropy_loss
        self.weight_q_loss = weight_q_loss
        self.lcb_coef = lcb_coef
        self.gamma = gamma
        self.max_q_backup = max_q_backup
        self.q_sample_eta = q_sample_eta
        self.grad_clip = grad_clip

    def _expectile_loss(self, diff, tau):
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * (diff ** 2)).mean()
    
    def set_weight_bc_loss(self, weight_bc_loss):
        self.weight_bc_loss = weight_bc_loss
    
    def update(self, batch_off, batch_on=None):
        if batch_on is not None:
            s, a, r, s_next = batch_on[0], batch_on[1], batch_on[2], batch_on[3]
            s_off, a_off = batch_off[0], batch_off[1]
        else:
            s, a, r, s_next = batch_off[0], batch_off[1], batch_off[2], batch_off[3]

        ##################
        """ Q Update """
        ##################
        with torch.no_grad():
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(s_next, repeats=10, dim=0)
                next_action_rpt = self.actor_target.target.sample_DDIM(next_state_rpt, eta=self.q_sample_eta).clamp(-self.abs_action_max, self.abs_action_max)
                q_next_rpt = self.critic_target.target.q_min(next_state_rpt, next_action_rpt)
                q_next = q_next_rpt.view(s_next.size(0), 10, -1).max(dim=1)[0]
            else:
                a_next = self.actor_target.target.sample_DDIM(s_next, eta=self.q_sample_eta).clamp(-self.abs_action_max, self.abs_action_max)
                q_next = self.critic_target.target.q_min(s_next, a_next)
        q_target = (r.unsqueeze(-1) + self.gamma * q_next).detach()
        
        q1_pred, q2_pred = self.critic(s, a)
        loss_q = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.grad_clip, norm_type=2)
        self.critic_optimizer.step()

        #######################
        """ Policy Training """
        #######################
        new_a0, new_a1 = self.actor.approximate_action(s, a)
        q_values_new_action_ensembles = self.critic.q_min(s, new_a0)
        mu = q_values_new_action_ensembles.mean(dim=1, keepdim=True)
        std = q_values_new_action_ensembles.std(dim=1, keepdim=True)
        q_values_new_action = mu - self.lcb_coef * std      
        L_q = -q_values_new_action.mean() / q_values_new_action_ensembles.abs().mean().detach()     

        if batch_on is not None:
            L_clone = self.actor.diffusion_loss(s_off, a_off)
        else:
            L_clone = self.actor.diffusion_loss(s, a)
        L_bc = L_clone.mean() / L_clone.abs().mean().detach()
        #L_bc = L_clone.mean()

        L_entropy = self.actor.entropy_loss(s, new_a0, new_a1)
        L_e = L_entropy.mean() / L_entropy.abs().mean().detach()   

        #######################
        """ Update Actor """
        #######################
        loss_pi = self.weight_bc_loss*L_bc + self.weight_q_loss * L_q + self.weight_entropy_loss * L_e
        self.optimizer_actor.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip, norm_type=2)
        self.optimizer_actor.step()
        if self.decay_lr == True:
            self.scheduler_lr_actor.step()
            self.scheduler_lr_critic.step()
        if self.ema_steps % self.ema_period == 0 and self.ema_steps > self.ema_begin_update:
            self.actor_target.soft_update()
            self.critic_target.soft_update()
        self.ema_steps += 1
 
        return L_clone.item(), q_values_new_action.mean().item(), L_entropy.mean().item(), loss_q.item()

    def train(self, replay_buffer, iterations, batch_size, tqdm_pos=0):
        metrics = {"Ld": [], "Lq": [], "Le": [], "loss_Q": []}
        for i in tqdm(range(iterations), position=tqdm_pos, leave=False):
            batch = replay_buffer.sample(batch_size)
            if batch is not None:
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
            sample_method: str = "greedy",
            N: int = 10, 
            eta: float = 0.1,
        ) -> torch.Tensor:
        B = s.size(0)
        s_rep = s.unsqueeze(1).expand(B, N, s.size(-1)).reshape(-1, s.size(-1))
        a_cand = self.actor.sample_DDIM(s_rep, eta=eta)
        q_cand = self.critic_target.target.q_min(s_rep, a_cand).mean(dim=-1).view(B, N)
        a_cand = a_cand.view(B, N, -1)
        if sample_method == "greedy":
            best_idx = torch.argmax(q_cand, dim=1)                     # (B,)
            a_best = a_cand[torch.arange(B), best_idx]  # (B, action_dim)
        elif sample_method == "EAS":
            probs  = torch.softmax(q_cand, dim=1)             # (B,N) 
            idx = torch.multinomial(probs, num_samples=1)     # (B,1)
            a_best = a_cand[torch.arange(B), idx.squeeze(-1)]
        elif sample_method == "bc":
            a_best = a_cand[torch.arange(B), torch.randint(0, N, (B,))]
        else:
            raise ValueError(f"Invalid sample method: {sample_method}")
        a_best = a_best.clamp(-self.abs_action_max, self.abs_action_max)
        return a_best

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:   
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', weights_only=True))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth', weights_only=True))
            self.actor_target.target.load_state_dict(self.actor.state_dict())
            self.critic_target.target.load_state_dict(self.critic.state_dict())
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', weights_only=True))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth', weights_only=True))
            self.actor_target.target.load_state_dict(self.actor.state_dict())
            self.critic_target.target.load_state_dict(self.critic.state_dict())
