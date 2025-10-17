
import numpy as np
import torch


def value_encode(x, min_x, max_x, n_bits, b=2):
    x = np.asarray(x, dtype=float)
    if np.any(x < min_x) or np.any(x > max_x):
        raise ValueError(f"Input values must be in [{min_x}, {max_x}].")
    if b < 2:
        raise ValueError("Base `b` must be at least 2.")
    
    N = x.shape[0]
    n_codes = b ** n_bits

    # compute bin indices 0…n_codes-1
    if max_x == min_x:
        bin_idx = np.zeros(N, dtype=int)
    else:
        bin_width = (max_x - min_x) / n_codes
        # floor into bins
        bin_idx = np.floor((x - min_x) / bin_width).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_codes - 1)

    # convert each index into n_bits base-b digits
    # highest-order digit first
    digits = np.zeros((N, n_bits), dtype=int)
    for pos in range(n_bits):
        power = b ** (n_bits - 1 - pos)
        digits[:, pos] = (bin_idx // power) % b

    return digits.reshape(-1)


def value_decode(y, min_x, max_x, n_bits, b=2):
    # ==== !!! should use round () ======
    #y = y.astype(int)
    y = np.round(y)
    if b < 2:
        raise ValueError("Base `b` must be at least 2.")
    
    # reshape into (N, n_bits) array of digits
    y = y.reshape(-1, n_bits)
    N = y.shape[0]
    n_codes = b ** n_bits

    if max_x == min_x:
        return np.full(N, min_x, dtype=float)

    bin_width = (max_x - min_x) / n_codes

    # recover bin indices from base-b digits
    powers = b ** np.arange(n_bits - 1, -1, -1)
    bin_idx = (y * powers).sum(axis=1)

    # map to the center of each bin
    x_recon = min_x + (bin_idx + 0.5) * bin_width
    return x_recon

def normalize(x, mi, ma):
    """
    Maps values in [mi, ma] to [-1, 1].
    """
    x = np.array(x)
    return np.clip(2 * (x - mi) / (ma - mi) - 1.0, -1.0, 1.0)

def denormalize(x, mi, ma):
    """
    Maps values in [-1, 1] back to [mi, ma].
    """
    x = np.array(x)
    return np.clip((x + 1.0) * (ma - mi) / 2.0 + mi, mi, ma)

class EnvInterface:
    def __init__(self, 
                 envParams,
                 discrete_state: bool = False,
                 n_bits_state: int = 8,
                 base_state: int = 2,
                 n_bits_action: int = 2,
                 base_action: int = 50,
                 ):
        self.n_users = envParams['N_user']
        self.bandwidth = envParams['B']
        self.len_window = envParams['LEN_window']
        self.discrete_state = discrete_state
        self.n_bits_state = n_bits_state
        self.base_state = base_state
        self.n_bits_action = n_bits_action
        self.base_action = base_action

        if self.discrete_state is False:
            self.state_dim = self.n_users # [u_i | i=0,1,...,N-1]
            self.action_dim = self.n_users # [r_i | i=0,1,...,N-1]
        else:
            self.state_dim = self.n_users * self.n_bits_state
            self.action_dim =  self.n_users * self.n_bits_action # [r_i | i=0,1,...,N-1]

        (self.mi_r, self.ma_r) = (0.0, self.bandwidth)

    def _find_closest(self, x, x_list):
        return [x_list[np.argmin(np.abs(x_list - x_))] for x_ in x]

    def _from_agent_action_to_env_action(self, action):
        # Convert normalized [-1, 1] actions to actual values
        action = np.clip(action, -1, 1)
        r_norm = action
        r = denormalize(r_norm, self.mi_r, self.ma_r)
        return r
    
    def _from_env_action_to_agent_action(self, action):
        r_norm = normalize(action, self.mi_r, self.ma_r)
        return r_norm

    def preprocess_action(self, action):
        a = self._from_env_action_to_agent_action(action)
        if self.discrete_state is False:
            return a
        else:
            return normalize(value_encode(a, -1.0, 1.0, self.n_bits_action, self.base_action), 0, self.base_action-1)
        
    def postprocess_action(self, action):
        if self.discrete_state is True:
            action = np.round(action)
            action = denormalize(action, 0, self.base_action-1)
            action = value_decode(action, -1.0, 1.0, self.n_bits_action, self.base_action)
        return  self._from_agent_action_to_env_action(action)
    
    def preprocess_state(self, state):
        s = 2*np.array(state) / self.len_window - 1.0
        if self.discrete_state is False:
            return s
        else:
            return normalize(value_encode(s, -1.0, 1.0, self.n_bits_state, self.base_state), 0, self.base_state-1)
        
    def preprocess_reward(self, reward, action):
        return 1.0-np.array(reward)
    
    def convert_from_env_data_to_agent_data(self, dataset):
        out = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}
        for s in dataset['observations']:
            out['observations'].append(self.preprocess_state(s))
        for a in dataset['actions']:
            out['actions'].append(self.preprocess_action(a))
        for r, a in zip(dataset['rewards'], dataset['actions']):
            out['rewards'].append(self.preprocess_reward(r, a))
        for s_next in dataset['next_observations']:
            out['next_observations'].append(self.preprocess_state(s_next))
        return out


