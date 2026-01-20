import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .Helpers import DiffusionSchedule, EMATarget
from .Actors import DiffusionPolicy
from .CriticsEsmb import EnsembleDoubleCritic

class DQL_Q_esmb_fast():
    """Optimized DiffusionQL with faster training and inference"""

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
        use_amp: bool = True,
        compile_model: bool = False,
        fast_inference_steps: int = 10,
        num_q_samples: int = 10,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.abs_action_max = abs_action_max
        self.use_amp = use_amp and device == "cuda"
        self.fast_inference_steps = fast_inference_steps
        self.num_q_samples = num_q_samples
        
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
        
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.actor = torch.compile(self.actor, mode='reduce-overhead')
                self.critic = torch.compile(self.critic, mode='reduce-overhead')
                print("Models compiled successfully")
            except Exception as e:
                print(f"Compilation failed: {e}, using eager mode")
        
        if self.use_amp:
            self.scaler_actor = GradScaler()
            self.scaler_critic = GradScaler()
        
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
    
    @torch.no_grad()
    def _compute_q_target(self, s_next, r):
        """Optimized Q-target computation"""
        if self.max_q_backup:
            B = s_next.size(0)
            s_expanded = s_next.unsqueeze(1).expand(-1, self.num_q_samples, -1).reshape(-1, s_next.size(-1))
            a_next = self.actor_target.target.sample_DDIM(s_expanded, eta=self.q_sample_eta)
            a_next = a_next.clamp_(-self.abs_action_max, self.abs_action_max)
            
            q_next = self.critic_target.target.q_min(s_expanded, a_next)
            q_next = q_next.view(B, self.num_q_samples, -1).max(dim=1)[0]
        else:
            a_next = self.actor_target.target.sample_DDIM(s_next, eta=self.q_sample_eta)
            a_next = a_next.clamp_(-self.abs_action_max, self.abs_action_max)
            q_next = self.critic_target.target.q_min(s_next, a_next)
        
        return (r.unsqueeze(-1) + self.gamma * q_next).detach()
    
    def update(self, batch_off, batch_on=None):
        if batch_on is not None:
            s, a, r, s_next = batch_on[0], batch_on[1], batch_on[2], batch_on[3]
            s_off, a_off = batch_off[0], batch_off[1]
        else:
            s, a, r, s_next = batch_off[0], batch_off[1], batch_off[2], batch_off[3]

        if self.use_amp:
            return self._update_amp(s, a, r, s_next, batch_on, s_off if batch_on else None, a_off if batch_on else None)
        else:
            return self._update_standard(s, a, r, s_next, batch_on, s_off if batch_on else None, a_off if batch_on else None)
    
    def _update_standard(self, s, a, r, s_next, batch_on, s_off, a_off):
        """Standard update without AMP"""
        q_target = self._compute_q_target(s_next, r)
        
        q1_pred, q2_pred = self.critic(s, a)
        loss_q = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip, norm_type=2)
        self.critic_optimizer.step()

        new_a0, new_a1 = self.actor.approximate_action(s, a)
        q_values_new_action_ensembles = self.critic.q_min(s, new_a0)
        
        q_mean = q_values_new_action_ensembles.mean(dim=1, keepdim=True)
        q_std = q_values_new_action_ensembles.std(dim=1, keepdim=True)
        q_values_new_action = q_mean - self.lcb_coef * q_std
        
        q_abs_mean = q_values_new_action_ensembles.abs().mean().detach()
        L_q = -q_values_new_action.mean() / q_abs_mean.clamp(min=1e-6)

        if batch_on is not None:
            L_clone = self.actor.diffusion_loss(s_off, a_off)
        else:
            L_clone = self.actor.diffusion_loss(s, a)
        L_bc = L_clone / L_clone.abs().clamp(min=1e-6).detach()

        L_entropy = self.actor.entropy_loss(s, new_a0, new_a1)
        L_e = L_entropy / L_entropy.abs().clamp(min=1e-6).detach()

        loss_pi = self.weight_bc_loss*L_bc + self.weight_q_loss * L_q + self.weight_entropy_loss * L_e
        self.optimizer_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip, norm_type=2)
        self.optimizer_actor.step()
        
        if self.decay_lr:
            self.scheduler_lr_actor.step()
            self.scheduler_lr_critic.step()
        
        if self.ema_steps % self.ema_period == 0 and self.ema_steps > self.ema_begin_update:
            self.actor_target.soft_update()
            self.critic_target.soft_update()
        self.ema_steps += 1
 
        return L_clone.item(), q_values_new_action.mean().item(), L_entropy.item(), loss_q.item()
    
    def _update_amp(self, s, a, r, s_next, batch_on, s_off, a_off):
        """Mixed precision update with autocast"""
        q_target = self._compute_q_target(s_next, r)
        
        with autocast():
            q1_pred, q2_pred = self.critic(s, a)
            loss_q = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler_critic.scale(loss_q).backward()
        self.scaler_critic.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip, norm_type=2)
        self.scaler_critic.step(self.critic_optimizer)
        self.scaler_critic.update()

        with autocast():
            new_a0, new_a1 = self.actor.approximate_action(s, a)
            q_values_new_action_ensembles = self.critic.q_min(s, new_a0)
            
            q_mean = q_values_new_action_ensembles.mean(dim=1, keepdim=True)
            q_std = q_values_new_action_ensembles.std(dim=1, keepdim=True)
            q_values_new_action = q_mean - self.lcb_coef * q_std
            
            q_abs_mean = q_values_new_action_ensembles.abs().mean().detach()
            L_q = -q_values_new_action.mean() / q_abs_mean.clamp(min=1e-6)

            if batch_on is not None:
                L_clone = self.actor.diffusion_loss(s_off, a_off)
            else:
                L_clone = self.actor.diffusion_loss(s, a)
            L_bc = L_clone / L_clone.abs().clamp(min=1e-6).detach()

            L_entropy = self.actor.entropy_loss(s, new_a0, new_a1)
            L_e = L_entropy / L_entropy.abs().clamp(min=1e-6).detach()

            loss_pi = self.weight_bc_loss*L_bc + self.weight_q_loss * L_q + self.weight_entropy_loss * L_e
        
        self.optimizer_actor.zero_grad(set_to_none=True)
        self.scaler_actor.scale(loss_pi).backward()
        self.scaler_actor.unscale_(self.optimizer_actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip, norm_type=2)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()
        
        if self.decay_lr:
            self.scheduler_lr_actor.step()
            self.scheduler_lr_critic.step()
        
        if self.ema_steps % self.ema_period == 0 and self.ema_steps > self.ema_begin_update:
            self.actor_target.soft_update()
            self.critic_target.soft_update()
        self.ema_steps += 1
 
        return L_clone.item(), q_values_new_action.mean().item(), L_entropy.item(), loss_q.item()

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
            eta: float = 0.0,
            use_fast_inference: bool = True,
        ) -> torch.Tensor:
        """Optimized sampling with fast inference mode"""
        B = s.size(0)
        
        if use_fast_inference and self.fast_inference_steps < self.sched.N:
            original_N = self.sched.N
            self.actor.schedule.N = self.fast_inference_steps
        
        s_rep = s.unsqueeze(1).expand(-1, N, -1).reshape(-1, s.size(-1))
        a_cand = self.actor.sample_DDIM(s_rep, eta=eta)
        
        if use_fast_inference and self.fast_inference_steps < original_N:
            self.actor.schedule.N = original_N
        
        a_cand = a_cand.view(B, N, -1)
        a_cand_flat = a_cand.reshape(-1, a_cand.size(-1))
        
        q_cand = self.critic_target.target.q_min(s_rep, a_cand_flat)
        q_cand = q_cand.mean(dim=-1).view(B, N)
        
        if sample_method == "greedy":
            best_idx = q_cand.argmax(dim=1)
            a_best = a_cand[torch.arange(B), best_idx]
        elif sample_method == "EAS":
            probs = F.softmax(q_cand, dim=1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            a_best = a_cand[torch.arange(B), idx]
        elif sample_method == "bc":
            idx = torch.randint(0, N, (B,), device=s.device)
            a_best = a_cand[torch.arange(B), idx]
        else:
            raise ValueError(f"Invalid sample method: {sample_method}")
        
        a_best = a_best.clamp_(-self.abs_action_max, self.abs_action_max)
        return a_best
    
    @torch.no_grad()
    def sample_fast(self, s: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Ultra-fast deterministic sampling for inference"""
        original_N = self.actor.schedule.N
        self.actor.schedule.N = self.fast_inference_steps
        
        a = self.actor.sample_DDIM(s, eta=0.0 if deterministic else 0.1)
        
        self.actor.schedule.N = original_N
        return a.clamp_(-self.abs_action_max, self.abs_action_max)

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

