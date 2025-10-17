import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Helpers import mlp, timestep_embedding
import numpy as np

class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, schedule, t_dim: int = 128, hidden_dim: int = 256, layer_norm: bool = False):
        super().__init__()
        self.s_dim, self.a_dim = state_dim, action_dim
        self.t_dim = t_dim
        self.schedule = schedule
        self.eps_net = mlp(state_dim + action_dim + self.t_dim, action_dim, 
                           hidden=(hidden_dim, hidden_dim, hidden_dim), p_dropout=0.05, layer_norm=layer_norm)
        self.ep_baseline = 0.0

    def forward(self, a_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.t_dim)
        return self.eps_net(torch.cat([a_t, s, emb], dim=-1))

    def diffusion_loss(self, s: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        B = a0.size(0)
        t = torch.randint(1, self.schedule.N+1, (B,), device=a0.device)
        alpha_bar = self.schedule.alpha_bar[t-1]   
        noise = torch.randn_like(a0)
        # q(a_t | a0)
        a_t = (alpha_bar.sqrt()[:, None] * a0 +
               (1 - alpha_bar).sqrt()[:, None] * noise)
        eps_pred = self(a_t, s, t)
        return F.mse_loss(eps_pred, noise)
    
    def entropy_loss(self, s: torch.Tensor, a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
        beta1 = self.schedule.beta[0]
        alpha_bar1 = self.schedule.alpha_bar[0]
        action_dim = a0.size(1)
        a0, a1 = a0.detach(), a1.detach()

        eps_a1 = self(a1, s, torch.ones(s.size(0), device=s.device, dtype=torch.long))
        #mu = (a1 - (beta1/(1 - alpha_bar1).sqrt()) * eps_a1) / alpha_bar1.sqrt()
        mu = (a1 - (1-alpha_bar1).sqrt() * eps_a1) / alpha_bar1.sqrt()
        sigma = (1 - alpha_bar1).sqrt()

        h1 = 0.5*(
            (a0 - mu).pow(2) / sigma.pow(2) + 
            torch.log(2.0*torch.pi*sigma.pow(2))
        )
        return h1.mean()

    
    def approximate_action(self,
                           s: torch.Tensor,
                           a0: torch.Tensor) -> torch.Tensor:
        B = a0.size(0)
        t = torch.randint(1, self.schedule.N+1, (B,), device=a0.device)
        alpha_bar = self.schedule.alpha_bar[t-1]
        noise = torch.randn_like(a0)
        # corrupt
        a_t = (alpha_bar.sqrt()[:, None] * a0 +
               (1 - alpha_bar).sqrt()[:, None] * noise)
        # predict noise
        eps_pred = self(a_t, s, t)
        # approximate clean action
        a0_hat = (a_t - (1 - alpha_bar).sqrt()[:, None] * eps_pred) / alpha_bar.sqrt()[:, None]

        alpha1 = self.schedule.alpha[0]
        a1_hat = alpha1.sqrt() * a0_hat + (1. - alpha1).sqrt() * noise
        return a0_hat, a1_hat
    
    def sample_DDPM(self, s: torch.Tensor) -> torch.Tensor:
        B = s.size(0)
        a = torch.randn(B, self.a_dim, device=s.device) * (1-self.schedule.alpha_bar[-1]).sqrt()
        noises = torch.randn(self.schedule.N, B, self.a_dim, device=s.device)
        for step in reversed(range(1, self.schedule.N + 1)):
            i = step - 1
            b_i = self.schedule.beta[i]
            a_i = self.schedule.alpha[i]
            ab_i = self.schedule.alpha_bar[i]

            t = torch.full((B,), step, device=s.device, dtype=torch.long)
            eps = self(a, s, t)
            mean = (a - b_i / (1 - ab_i).sqrt() * eps) / a_i.sqrt()

            noise = noises[i] if step > 1 else 0.0
            a = mean + b_i.sqrt() * noise
        return a
    
    def sample_DDIM(self, s: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        B = s.size(0)
        #eta = np.clip(eta, 0.0, 1.0)
        #a = torch.randn(B, self.a_dim, device=s.device) * (1-self.schedule.alpha_bar[-1]).sqrt()
        a = torch.randn(B, self.a_dim, device=s.device) * eta * (1-self.schedule.alpha_bar[-1]).sqrt()
        noises = torch.randn(self.schedule.N, B, self.a_dim, device=s.device)
        for step in reversed(range(1, self.schedule.N + 1)):
            alpha = self.schedule.alpha_bar[step - 1]
            alpha_prev = self.schedule.alpha_bar[step-2] if step > 1 else torch.tensor(1.0, device=s.device)

            t = torch.full((B,), step, device=s.device, dtype=torch.long)
            eps = self(a, s, t)

            sigma_t = eta * ((1 - alpha_prev) / (1 - alpha)).sqrt() * (1 - alpha/alpha_prev).sqrt()
            #sigma_t = eta *(1 - alpha/alpha_prev).sqrt()
            x0_pred = alpha_prev.sqrt() * (a - (1-alpha).sqrt() * eps) / alpha.sqrt()
            direct_point_x = (1 - alpha_prev - sigma_t.pow(2)).clamp(0.0).sqrt() * eps
            noise_term = sigma_t * noises[step - 1]
            a = x0_pred + direct_point_x + noise_term
        return a

    