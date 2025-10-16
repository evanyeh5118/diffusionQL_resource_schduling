import numpy as np
import matplotlib.pyplot as plt
import os

from Agents.DrlLibs.DRL_config import (
    get_algorithm_config, 
    get_training_config,
)
from Agents.DrlLibs.training import create_environment

def evaluate_drl_agent(model, env, n_steps, algorithm_name, deterministic=True):    
    mode_str = "Deterministic" if deterministic else "Stochastic (with exploration)"
    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm_name} Agent for {n_steps} steps")
    print(f"Mode: {mode_str}")
    print(f"{'='*60}")
    
    packet_loss_rates = []
    rewards = []
    actions = []
    observations = []

    # Reset environment
    env.reset()
    # Run for n_steps
    for step in range(n_steps):
        obs = env.observe()
        action, _ = model.predict(obs, deterministic=deterministic)
        step_result = env.step(action)
            
        # Handle different step return formats
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        # Store reward
        rewards.append(reward)
        actions.append(env.unwrapped._from_dl_action_to_env_action(action))
        observations.append(obs*env.len_window)
        # Extract packet loss rate
        if isinstance(info, dict) and 'total_packet_loss_rate' in info:
            packet_loss_rates.append(info['total_packet_loss_rate'])
        else:
            packet_loss_rates.append(0.0)
    
    
    # Calculate statistics
    avg_packet_loss = np.mean(packet_loss_rates)
    std_packet_loss = np.std(packet_loss_rates)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    results = {
        'algorithm': algorithm_name,
        'n_steps': n_steps,
        'avg_packet_loss': avg_packet_loss,
        'std_packet_loss': std_packet_loss,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'packet_loss_rates': packet_loss_rates,
        'rewards': rewards,
        'actions': actions,
        'observations': observations
    }
    
    print(f"Evaluation Results:")
    print(f"  Steps Completed: {n_steps}")
    print(f"  Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"  Average Packet Loss: {avg_packet_loss:.4f} ± {std_packet_loss:.4f}")
    
    return results


def load_and_evaluate(simParams, simEnv, load_path, algorithm_name, 
                     n_steps, obvMode="perfect", episode_timesteps=1000, 
                     deterministic=True):
    """
    Load a trained model and evaluate it.
    
    Parameters:
    -----------
    simParams : dict
        Simulation parameters
    simEnv : Environment
        Simulation environment
    load_path : str
        Path to load the trained model from
    algorithm_name : str
        Name of the algorithm
    n_steps : int
        Number of steps to run for evaluation
    obvMode : str, optional
        Observation mode (default: "perfect")
    episode_timesteps : int, optional
        Number of timesteps per episode (default: 1000)
    deterministic : bool, optional
        If True, use deterministic actions (no exploration noise)
        If False, use stochastic actions (with exploration, like during training)
        (default: True)
    
    Returns:
    --------
    model : trained model
        The loaded model
    eval_results : dict
        Evaluation results
    """
    
    print(f"Loading {algorithm_name} model from {load_path}")
    
    # Create environment
    env = create_environment(simParams, simEnv, obvMode, episode_timesteps)
    
    # Get algorithm class
    config = get_algorithm_config(algorithm_name, env)
    algorithm_class = config["class"]
    
    # Load model
    model = algorithm_class.load(load_path)
    
    # Evaluate
    eval_results = evaluate_drl_agent(model, env, n_steps, algorithm_name, deterministic=deterministic)
    
    env.close()
    return model, eval_results
