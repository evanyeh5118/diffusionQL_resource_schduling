import numpy as np
import matplotlib.pyplot as plt
import torch

def create_toy_gaussian_dataset(
        N_samples: int = 5000, 
    ):

    # Mixture component means and per-dimension std
    mus = np.array([
        [ 0.0,  0.8],
        [ 0.8,  0.0],
        [ 0.0, -0.8],
        [-0.8,  0.0],
    ])
    sigma = 0.05

    # Reward means for each component
    reward_means = np.array([1.0, 0.0, -1.5, 5.0])
    reward_std   = 0.5

    # 1) Sample component indices uniformly
    component_ids = np.random.randint(0, 4, size=N_samples)

    # 2) Sample actions from the corresponding Gaussian
    actionsRecord = mus[component_ids] + sigma * np.random.randn(N_samples, 2)

    # 3) Draw rewards based on which component each action came from
    rewardRecord = reward_means[component_ids] + reward_std * np.random.randn(N_samples)

    # 4) States and next-states are always zero (bandit has no dynamics)
    sRecord     = np.zeros((N_samples, 1))
    sNextRecord = np.zeros((N_samples, 1))

    dataset = {
        'observations': sRecord,
        'actions': actionsRecord,
        'rewards': rewardRecord,
        'next_observations': sNextRecord,
    }
    return dataset


def eval(agent, N_eval = 100, N_sample=10, eta=0.1, sample_method="greedy", figsize=(3, 3)):
    # Generate evaluation states (all zeros for bandit)
    dataset = create_toy_gaussian_dataset(N_eval)

    s_eval = np.zeros((N_eval, 1), dtype=np.float32)
    s_eval_tensor = torch.as_tensor(s_eval, dtype=torch.float32).to(agent.device)

    # Sample actions from the policy for these states
    with torch.no_grad():
        a_eval = agent.sample(
                s_eval_tensor, sample_method=sample_method, N=N_sample, eta=eta)
        a_eval = a_eval.cpu().detach().numpy()

    # Plot the training actions and the evaluated actions
    plt.figure(figsize=figsize)
    plt.scatter(dataset['actions'][:, 0], dataset['actions'][:, 1], alpha=0.3, label="Training actions")
    plt.scatter(a_eval[:, 0], a_eval[:, 1], alpha=0.8, color='red', label="Policy actions")
    plt.xlabel("Action dim 1")
    plt.ylabel("Action dim 2")
    plt.title("Policy Actions vs Training Data")
    plt.legend()
    plt.grid(True)
    plt.show()

    a_train = dataset['actions']
    return a_train, a_eval