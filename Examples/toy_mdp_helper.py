"""
Toy MDP Helper Functions

This module provides utility functions for working with Markov Decision Processes (MDPs)
in the context of diffusion-based reinforcement learning. It includes functions for
data normalization, MDP generation, policy evaluation, and dataset creation.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import warnings

# Import local modules
try:
    from Agents.ModelBasedSolvers import MdpKernel
    from Environment.Helpers.TrafficGenerator import generate_next_state
except ImportError as e:
    warnings.warn(f"Some dependencies could not be imported: {e}")


def normalize_data(data: Union[float, np.ndarray, torch.Tensor], 
                  min_val: float, 
                  max_val: float) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Normalize data from [min_val, max_val] to [-1, 1].
    
    Args:
        data: Input data to normalize
        min_val: Minimum value of the original range
        max_val: Maximum value of the original range
        
    Returns:
        Normalized data in range [-1, 1]
        
    Raises:
        ValueError: If min_val >= max_val
    """
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    
    if isinstance(data, torch.Tensor):
        return 2 * (data - min_val) / (max_val - min_val) - 1
    else:
        return 2 * (np.asarray(data) - min_val) / (max_val - min_val) - 1


def denormalize_data(data: Union[float, np.ndarray, torch.Tensor], 
                    min_val: float, 
                    max_val: float) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Denormalize data from [-1, 1] back to [min_val, max_val].
    
    Args:
        data: Input data to denormalize
        min_val: Minimum value of the target range
        max_val: Maximum value of the target range
        
    Returns:
        Denormalized data in range [min_val, max_val]
        
    Raises:
        ValueError: If min_val >= max_val
    """
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    
    if isinstance(data, torch.Tensor):
        return (data + 1) / 2 * (max_val - min_val) + min_val
    else:
        return (np.asarray(data) + 1) / 2 * (max_val - min_val) + min_val


def generate_transition_and_reward(N_s: int, 
                                 N_a: int, 
                                 seed: Optional[int] = None,
                                 reward_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random transition probabilities and rewards for an MDP.
    
    Args:
        N_s: Number of states
        N_a: Number of actions
        seed: Random seed for reproducibility
        reward_range: Tuple of (min_reward, max_reward) for reward generation
        
    Returns:
        Tuple of (P, R) where:
            P: Transition probability matrix of shape (N_s, N_s, N_a)
            R: Reward matrix of shape (N_s, N_a)
            
    Raises:
        ValueError: If N_s or N_a is not positive
    """
    if N_s <= 0 or N_a <= 0:
        raise ValueError("N_s and N_a must be positive integers")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random transition probabilities
    P = np.random.rand(N_s, N_a, N_s)
    # Normalize over next-states axis so that ∑_s' P(s'|s,a) = 1
    P /= P.sum(axis=2, keepdims=True)
    
    # Add small epsilon to avoid zero probabilities
    epsilon = 1e-10
    P += epsilon
    P /= P.sum(axis=2, keepdims=True)
    
    # Generate random rewards within specified range
    min_reward, max_reward = reward_range
    R = np.random.uniform(min_reward, max_reward, (N_s, N_a))
    
    # Transpose P to match expected format (N_s, N_s, N_a)
    P = np.transpose(P, (0, 2, 1))
    
    return P, R


def evaluate_diffusionQ(agent: Any, 
                       R: np.ndarray, 
                       P: np.ndarray, 
                       N_s: int, 
                       N_a: int, 
                       N_iter: int = 100, 
                       sample_method: str = "greedy",
                       N_sample: int = 20,
                       eta: float = 0.1,
                       initial_state: Optional[int] = None) -> Tuple[float, Dict[str, List]]:
    """
    Evaluate a DiffusionQ agent on an MDP environment.
    
    Args:
        diffusionQ: DiffusionQ agent to evaluate
        R: Reward matrix of shape (N_s, N_a)
        P: Transition probability matrix of shape (N_s, N_s, N_a)
        N_s: Number of states
        N_a: Number of actions
        N_iter: Number of evaluation iterations
        sample_method: Sampling method for the agent
        initial_state: Initial state (if None, random state is chosen)
        
    Returns:
        Tuple of (mean_reward, metrics) where metrics contains trajectory data
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If agent evaluation fails
    """
    if N_iter <= 0:
        raise ValueError("N_iter must be positive")

    
    metrics = {
        'observations': [], 
        'actions': [], 
        'rewards': [], 
        'next_observations': []
    }
    
    # Set initial state
    if initial_state is not None:
        if not (0 <= initial_state < N_s):
            raise ValueError(f"initial_state must be in range [0, {N_s})")
        s = initial_state
    else:
        s = np.random.randint(0, N_s)
    
    try:
        for _ in tqdm(range(N_iter), desc="Evaluating DiffusionQ", leave=False):
            # Prepare state for agent
            s_DQ = torch.as_tensor([normalize_data(s, 0, N_s-1)], 
                                 dtype=torch.float32).to(agent.device)
            
            # Sample action from agent
            a_DQ = agent.sample(s_DQ, N=N_sample, eta=eta, sample_method=sample_method)
            a_DQ = a_DQ.cpu().detach().numpy()[0][0]
            
            # Denormalize action
            a = denormalize_data(a_DQ, 0, N_a-1).astype(int)
            a = np.clip(a, 0, N_a-1)  # Ensure action is within bounds
            
            # Get reward and next state
            r = R[s, a]
            s_next = generate_next_state(P[:, :, a], s)
            
            # Update state
            s = s_next
            
            # Record metrics
            metrics['observations'].append([s])
            metrics['actions'].append([a])
            metrics['rewards'].append([r])
            metrics['next_observations'].append([s_next])
            
    except Exception as e:
        raise RuntimeError(f"Error during DiffusionQ evaluation: {e}")
    
    mean_reward = np.mean(metrics['rewards'])
    return mean_reward, metrics


def evaluate_deterministic_policy(policyDeter: List[int], 
                                R: np.ndarray, 
                                P: Optional[np.ndarray], 
                                sRecord: Optional[List[int]], 
                                N_s: int, 
                                N_a: int, 
                                N_iter: int = 100) -> Tuple[float, Dict[str, List]]:
    """
    Evaluate a deterministic policy on an MDP environment.
    
    Args:
        policyDeter: Deterministic policy as list of actions for each state
        R: Reward matrix of shape (N_s, N_a)
        P: Transition probability matrix of shape (N_s, N_s, N_a) or None
        sRecord: Pre-recorded state sequence or None
        N_s: Number of states
        N_a: Number of actions
        N_iter: Number of evaluation iterations
        
    Returns:
        Tuple of (mean_reward, metrics) where metrics contains trajectory data
        
    Raises:
        ValueError: If inputs are invalid
    """
    if len(policyDeter) != N_s:
        raise ValueError(f"policyDeter length ({len(policyDeter)}) must equal N_s ({N_s})")
    
    if N_iter <= 0:
        raise ValueError("N_iter must be positive")
    
    if P is None and sRecord is None:
        raise ValueError("Either P or sRecord must be provided")
    
    if sRecord is not None and len(sRecord) < N_iter:
        raise ValueError(f"sRecord length ({len(sRecord)}) must be at least N_iter ({N_iter})")
    
    metrics = {
        's_record': [], 
        'a_record': [], 
        'r_record': [], 
        's_next_record': []
    }
    
    # Set initial state
    if P is None:
        s = sRecord[0]
    else:
        s = np.random.randint(0, N_s)
    
    try:
        for i in tqdm(range(N_iter), desc="Evaluating deterministic policy", leave=False):
            if P is None:
                s = sRecord[i]
            else:
                # Get action for current state
                a = policyDeter[s]
                if not (0 <= a < N_a):
                    raise ValueError(f"Invalid action {a} for state {s}")
                
                # Get next state
                s_next = generate_next_state(P[:, :, a], s)
            
            # Get action and reward
            a = policyDeter[s]
            r = R[s, a]
            
            # Record reward
            metrics['r_record'].append(r)
            
            # Record additional metrics if P is provided
            if P is not None:
                metrics['s_record'].append([normalize_data(s, 0, N_s-1)])
                metrics['a_record'].append([normalize_data(a, 0, N_a-1)])
                metrics['s_next_record'].append([normalize_data(s_next, 0, N_s-1)])
                s = s_next
                
    except Exception as e:
        raise RuntimeError(f"Error during deterministic policy evaluation: {e}")
    
    mean_reward = np.mean(metrics['r_record'])
    return mean_reward, metrics


def extract_deterministic_policy(agent: Any, 
                               N_s: int, 
                               N_a: int, 
                               sample_method: str = "greedy",
                               eta: float = 0.1,
                               N_sample: int = 20,
                               N_sampling: int = 100) -> Tuple[List[int], np.ndarray]:
    if N_s <= 0 or N_a <= 0:
        raise ValueError("N_s and N_a must be positive")
    
    if N_sample <= 0:
        raise ValueError("N_sample must be positive")
    
    if N_sampling <= 0:
        raise ValueError("N_sampling must be positive")
    
    policy_deterministic = []
    frequency_heatmap = np.zeros((N_s, N_a))
    
    try:
        for s in tqdm(range(N_s), desc="Extracting deterministic policy", leave=False):
            # Prepare state for agent
            s_DQ = torch.as_tensor([normalize_data(s, 0, N_s-1)], 
                                 dtype=torch.float32).to(agent.device)
                             
            # Sample action from agent
            s_DQ_batch = s_DQ.repeat(N_sampling, 1)
            a_DQ = agent.sample(s_DQ_batch , N=N_sample, eta=eta, sample_method=sample_method)           

            a_DQ = a_DQ.cpu().detach().numpy()
            
            # Denormalize and clip action
            a = denormalize_data(a_DQ, 0, N_a-1).astype(int)
            a = np.clip(a, 0, N_a-1)

            # Count frequency of each action for this state
            for a_i in a:
                frequency_heatmap[s, a_i] += 1
            
            # Normalize frequency to get probability distribution
            frequency_heatmap[s, :] /= N_sampling
            
            # Get most frequent action (deterministic policy)
            most_frequent_action = np.argmax(frequency_heatmap[s, :])
            policy_deterministic.append(most_frequent_action)
            
    except Exception as e:
        raise RuntimeError(f"Error during policy extraction: {e}")
    
    return policy_deterministic, frequency_heatmap


def visualize_policy_heatmap(frequency_heatmap: np.ndarray, 
                           policy_deterministic: List[int],
                           title: str = "Policy Frequency Heat Map",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (8, 6)) -> None:

    if frequency_heatmap.ndim != 2:
        raise ValueError("frequency_heatmap must be 2-dimensional")
    
    N_s, N_a = frequency_heatmap.shape
    
    if len(policy_deterministic) != N_s:
        raise ValueError("policy_deterministic length must match N_s")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heat map
        im = ax.imshow(frequency_heatmap, cmap='Blues', aspect='auto')
        
        # Set labels and title
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Action', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_xticks(range(N_a))
        ax.set_yticks(range(N_s))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability', fontsize=10)
        
        # Add text annotations
        for i in range(N_s):
            for j in range(N_a):
                prob = frequency_heatmap[i, j]
                # Highlight the deterministic policy action
                color = "red" if j == policy_deterministic[i] else ("black" if prob < 0.5 else "white")
                weight = "bold" if j == policy_deterministic[i] else "normal"
                
                text = ax.text(j, i, f'{prob:.3f}',
                              ha="center", va="center", 
                              color=color, fontweight=weight, fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("matplotlib not available. Printing heat map as text:")
        print(frequency_heatmap)
        print(f"Deterministic policy: {policy_deterministic}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MultipleLocator
from typing import List, Optional, Tuple

def visualize_policy_heatmap_continuous(
        frequency_heatmaps: List[np.ndarray],
        policy_deterministic_list: List[List[int]],
        titles: Optional[List[str]] = None,
        main_title: str = "Policy Frequency Heat Maps (Continuous)",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        interpolation: str = "bilinear",
        xtick_step: int = 2,
        ytick_step: int = 2,
        gamma: float = 1.5,
        bbox_to_anchor: Tuple[float, float] = (0.5, -0.1),
        cbar_pos: Tuple[float, float, float, float] = (0.92, 0.1, 0.02, 0.8),
        font_scale: float = 1.0
    ) -> None:
        """
        Display multiple policy frequency heatmaps side by side with a shared colorbar and legend.
        """
        n = len(frequency_heatmaps)
        if titles is None:
            titles = [f"Heatmap {i+1}" for i in range(n)]
        if len(policy_deterministic_list) != n or len(titles) != n:
            raise ValueError("Length of frequency_heatmaps, policy_deterministic_list, and titles must match")

        # Create subplots
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
        if n == 1:
            axes = [axes]

        # Determine global norm for color mapping
        all_values = np.concatenate([hm.flatten() for hm in frequency_heatmaps])
        norm = PowerNorm(gamma=gamma, vmin=all_values.min() + 1e-8, vmax=all_values.max())

        # Plot each heatmap
        for i, (ax, freq, policy, title) in enumerate(zip(axes, frequency_heatmaps, policy_deterministic_list, titles)):
            N_s, N_a = freq.shape
            im = ax.imshow(
                freq,
                cmap='Blues',
                aspect='auto',
                interpolation=interpolation,
                origin='lower',
                norm=norm
            )
            # Overlay deterministic actions
            states = np.arange(N_s)
            actions = np.array(policy)
            sc = ax.scatter(actions, states, c='red', s=60, marker='o')

            ax.set_title(title, fontsize=12*font_scale, fontweight='bold')
            #ax.set_xlabel('Action', fontsize=10*font_scale)
            import string
            letter = string.ascii_lowercase[i]          # 'a', 'b', 'c', …
            ax.set_xlabel(f'Action\n({letter})',
                        fontsize=10*font_scale,
                        labelpad=0)   # tweak labelpad if you want more/less space
            ax.xaxis.set_major_locator(MultipleLocator(xtick_step*font_scale))
            ax.yaxis.set_major_locator(MultipleLocator(ytick_step*font_scale))
            ax.grid(True, which='both', color='white', alpha=0.3, linestyle='-')

            # --- new: add subfigure letter (a), (b), … just below the xlabel
           

        # Common labels
        axes[0].set_ylabel('State', fontsize=10*font_scale)
        fig.suptitle(main_title, fontsize=14*font_scale, fontweight='bold')

        # Shared colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Probability', fontsize=10*font_scale)
        cbar.ax.set_position(cbar_pos)

        # Shared legend for deterministic dots
        fig.legend([sc], ['Deterministic action'], loc='lower center', ncol=1,
                framealpha=0.7, fontsize=10*font_scale, bbox_to_anchor=bbox_to_anchor)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()
       


def get_mdp_policy(P: np.ndarray, 
                  R: np.ndarray, 
                  N_s: int, 
                  N_a: int,
                  gamma: float = 0.99,
                  theta: float = 1e-10) -> Tuple[np.ndarray, List[int]]:
    if not (0 < gamma < 1):
        raise ValueError("gamma must be in range (0, 1)")
    
    if theta <= 0:
        raise ValueError("theta must be positive")
    
    try:
        mdpKernel = MdpKernel()
        mdpKernel.N_states = N_s
        mdpKernel.N_actions = N_a
        mdpKernel.transitionTable = P
        mdpKernel.rewardTable = -R  # Note: negative rewards for maximization
        
        V, policyDeter = mdpKernel.optimize_policy(
            mode="deterministic", 
            gamma=gamma, 
            theta=theta
        )
        
        return V, policyDeter
        
    except Exception as e:
        raise RuntimeError(f"Error solving MDP: {e}")


def generate_dataset(len_dataset: int, 
                    N_s: int, 
                    N_a: int, 
                    seed: int = 995,
                    reward_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[Dict[str, List], List[int], np.ndarray, np.ndarray]:
    """
    Generate a complete dataset for training and evaluation.
    
    Args:
        len_dataset: Length of the dataset to generate
        N_s: Number of states
        N_a: Number of actions
        seed: Random seed for reproducibility
        reward_range: Range for reward generation
        
    Returns:
        Tuple of (trajectory, policyDeter, P, R) where:
            trajectory: Dictionary containing state, action, reward, and next_state sequences
            policyDeter: Optimal deterministic policy
            P: Transition probability matrix
            R: Reward matrix
            
    Raises:
        ValueError: If inputs are invalid
    """
    if len_dataset <= 0:
        raise ValueError("len_dataset must be positive")
    
    # Generate MDP
    P, R = generate_transition_and_reward(N_s, N_a, seed=seed, reward_range=reward_range)
    
    # Solve for optimal policy
    _, policyDeter = get_mdp_policy(P, R, N_s, N_a)
    
    # Generate trajectory using optimal policy
    _, trajectory = evaluate_deterministic_policy(
        policyDeter, R, P, sRecord=None, N_s=N_s, N_a=N_a, N_iter=len_dataset
    )
    
    # Format trajectory
    trajectory = {
        'observations': trajectory['s_record'], 
        'actions': trajectory['a_record'], 
        'rewards': trajectory['r_record'], 
        'next_observations': trajectory['s_next_record']
    }
    
    return trajectory, policyDeter, P, R


def validate_mdp_parameters(P: np.ndarray, R: np.ndarray, N_s: int, N_a: int) -> bool:
    """
    Validate MDP parameters for consistency.
    
    Args:
        P: Transition probability matrix
        R: Reward matrix
        N_s: Number of states
        N_a: Number of actions
        
    Returns:
        True if parameters are valid, False otherwise
    """
    try:
        # Check shapes
        if P.shape != (N_s, N_s, N_a):
            return False
        
        if R.shape != (N_s, N_a):
            return False
        
        # Check transition probabilities sum to 1
        for s in range(N_s):
            for a in range(N_a):
                if not np.isclose(P[s, :, a].sum(), 1.0, atol=1e-6):
                    return False
        
        # Check for valid probabilities
        if np.any(P < 0) or np.any(P > 1):
            return False
        
        return True
        
    except Exception:
        return False


def compute_policy_performance(policy: List[int], 
                             R: np.ndarray, 
                             P: np.ndarray, 
                             gamma: float = 0.99,
                             max_iter: int = 1000) -> float:
    """
    Compute the performance (expected discounted return) of a policy.
    
    Args:
        policy: Policy as list of actions
        R: Reward matrix
        P: Transition probability matrix
        gamma: Discount factor
        max_iter: Maximum iterations for value iteration
        
    Returns:
        Expected discounted return of the policy
    """
    N_s = len(policy)
    V = np.zeros(N_s)
    
    for _ in range(max_iter):
        V_new = np.zeros(N_s)
        for s in range(N_s):
            a = policy[s]
            V_new[s] = R[s, a] + gamma * np.sum(P[s, :, a] * V)
        
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
            
        V = V_new
    
    return np.mean(V)