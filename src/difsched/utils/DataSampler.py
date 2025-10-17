import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
import random

import torch


class ReplayBuffer:
    def __init__(self, capacity, envInterface=None, device="cpu"):
        self.device = device
        self.envInterface = envInterface
        self.capacity = capacity

        # storage arrays for (u, a, r, u')
        self.u_buf     = np.empty((capacity,), dtype=object)
        self.a_buf     = np.empty((capacity,), dtype=object)
        self.r_buf     = np.empty((capacity,), dtype=object)
        self.u2_buf    = np.empty((capacity,), dtype=object)
        self.next_idx  = 0
        self.size      = 0

    def __len__(self):
        return self.size

    def add(self, data):
        """Add a single transition (or batch of transitions)."""
        # allow batch add
        if self.envInterface is not None:
            data_agent = self.envInterface.convert_from_env_data_to_agent_data(data)
            u, a, r, u2 = data_agent['observations'], data_agent['actions'], data_agent['rewards'], data_agent['next_observations']
        else:
            u, a, r, u2 = data['observations'], data['actions'], data['rewards'], data['next_observations']
        batch_n = np.array(u).shape[0]
        for i in range(batch_n):
            idx = self.next_idx
            self.u_buf[idx]   = u[i]
            self.a_buf[idx]   = a[i]
            self.r_buf[idx]   = r[i]
            self.u2_buf[idx]  = u2[i]
            self.next_idx = (self.next_idx + 1) % self.capacity
            self.size     = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch with importance-sampling weights."""
        if self.size == 0:
            return None
        batch_size = min(batch_size, self.size)
        data_idx = np.random.randint(0, self.size, size=(batch_size,))
        batch_u = np.stack(self.u_buf[data_idx])
        batch_a = np.stack(self.a_buf[data_idx])
        batch_r = np.stack(self.r_buf[data_idx])
        batch_u2 = np.stack(self.u2_buf[data_idx])
        data = (
            torch.from_numpy(batch_u.astype(np.float32)).to(self.device),
            torch.from_numpy(batch_a.astype(np.float32)).to(self.device),
            torch.from_numpy(batch_r.astype(np.float32)).to(self.device),
            torch.from_numpy(batch_u2.astype(np.float32)).to(self.device)
        )
        return data


class ReplayBufferHybrid:
    def __init__(self, capacity, envInterface=None, device="cpu"):
        self.device = device
        self.envInterface = envInterface
        self.online_buffer = ReplayBuffer(capacity, envInterface=envInterface, device=device)
        self.offline_buffer = ReplayBuffer(capacity, envInterface=envInterface, device=device)
        self.sample_ratio = 0.5
    
    def addOnline(self, data):
        self.online_buffer.add(data)

    def addOffline(self, data):
        self.offline_buffer.add(data)
    
    def sample(self, batch_size):
        
        if len(self.offline_buffer) == 0:
            samples = self.online_buffer.sample(batch_size)
        elif len(self.online_buffer) == 0:
            samples = self.offline_buffer.sample(batch_size)
        else:
            samples_off = self.offline_buffer.sample(int(batch_size*self.sample_ratio))
            samples_on = self.online_buffer.sample(max(1, int(batch_size*(1-self.sample_ratio))))
            samples = (
                torch.cat((samples_off[0], samples_on[0]), dim=0), 
                torch.cat((samples_off[1], samples_on[1]), dim=0), 
                torch.cat((samples_off[2], samples_on[2]), dim=0), 
                torch.cat((samples_off[3], samples_on[3]), dim=0)
            )
        return samples
    
    def sample_online(self, batch_size):
        samples = self.online_buffer.sample(batch_size)
        return samples
    
    def sample_offline(self, batch_size):
        samples = self.offline_buffer.sample(batch_size)
        return samples
    
    def set_sample_ratio(self, ratio):
        self.sample_ratio = ratio

    def __len__(self):
        return len(self.offline_buffer) + len(self.online_buffer)
    
#====================================================================

