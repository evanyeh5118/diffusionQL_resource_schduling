import numpy as np
import d3rlpy

def from_env_to_d3rlpy_dataset(dataset_off, expConfig, episode_length=1000):
    """
    Convert dataset_off to d3rlpy-compatible format.
    
    dataset_off contains:
        - observations: list of numpy arrays
        - actions: list of tuples (w, r, M, alpha)
        - rewards: list of floats
        - next_observations: list of numpy arrays
    
    episode_length: number of transitions per episode (default: 1000)
    """
    T = len(dataset_off['observations'])
    
    
    observations = np.array(dataset_off['observations'], dtype=np.float32)
    observations = observations / (expConfig['LEN_window'] + 1e-8)
    
    n_users = observations.shape[1]
    action_dim = 2 * n_users + 2
    actions = np.zeros((T, action_dim), dtype=np.float32)
    
    for i, action_tuple in enumerate(dataset_off['actions']):
        w, r, M, alpha = action_tuple
        
        w_dl = (np.array(w, dtype=np.float32) * 2.0) - 1.0
        
        r_dl = (np.array(r, dtype=np.float32) / expConfig['LEN_window']) * 2.0 - 1.0
        
        M_dl = ((M - 1) / (10 - 1)) * 2.0 - 1.0
        
        alpha_dl = alpha * 2.0 - 1.0
        
        actions[i, :n_users] = w_dl
        actions[i, n_users:2*n_users] = r_dl
        actions[i, 2*n_users] = M_dl
        actions[i, 2*n_users + 1] = alpha_dl
    
    rewards = 1-np.array(dataset_off['rewards'], dtype=np.float32)
    
    terminals = np.zeros(T, dtype=np.float32)
    for i in range(episode_length - 1, T, episode_length):
        terminals[i] = 1.0
    
    timeouts = np.zeros(T, dtype=np.float32)
    
    assert observations.shape[0] == actions.shape[0] == rewards.shape[0] == terminals.shape[0]
    assert rewards.ndim == 1 and terminals.ndim == 1
    
    print(f"Observations shape: {observations.shape}")
    print(f"Number of episodes: {int(np.sum(terminals))}")

    # Check if observations are in [0, 1]
    assert np.all(observations >= 0.0) and np.all(observations <= 1.0), "Observations not in [0, 1]"
    # Check if actions are in [-1, 1]
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0), "Actions not in [-1, 1]"
    # Check if rewards are in [0, 1]
    assert np.all(rewards >= 0.0) and np.all(rewards <= 1.0), "Rewards not in [0, 1]"


    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )
    
    return dataset