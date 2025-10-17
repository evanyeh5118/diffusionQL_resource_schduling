#!/usr/bin/env python3
"""
Demo script for continuous policy visualization with interpolation.

This script demonstrates the new continuous visualization functions that present
the state-action space with proper interpolation instead of discrete grid-based visualization.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import sys
sys.path.append("..")

from Helpers.DataSampler import ReplayBuffer
from toy_mdp_helper import (
    generate_dataset, 
    evaluate_diffusionQ, 
    extract_deterministic_policy, 
    visualize_policy_heatmap,
    visualize_continuous_policy,
    visualize_continuous_policy_3d,
    visualize_policy_comparison
)

def main():
    """Main demonstration function."""
    
    print("=== Continuous Policy Visualization Demo ===\n")
    
    # Setup hyperparameters
    hyperparams = {
        'N_diffusion_steps': 30,
        'schedule_type': "vp",
        'abs_action_max': 1.0,
        'gamma': 0.99,
        'lr': 5e-3,
        'decay_lr': False,
        'weight_decay': 0.001,
        'num_critics': 8,
        'lcb_coef': 0.9,
        'q_sample_eta': 0.5,
        'weight_entropy_loss': 0.1,
        'weight_q_loss': 0.5,
        'approximate_action': True,
        'ema_tau': 0.005,
        'ema_period': 20,
        'ema_begin_update': 1000,
        'layer_norm': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # MDP parameters
    N_s = 10
    N_a = 10
    LEN_dataset = 1000
    
    print(f"Device: {hyperparams['device']}")
    print(f"MDP: {N_s} states, {N_a} actions")
    
    # Generate dataset and expert policy
    print("\nGenerating dataset and expert policy...")
    dataset, policy_mdp, P, R = generate_dataset(100, N_s, N_a, seed=995)
    
    # Load agent
    from DiffusionQL.DQL_Q_esmb import DQL_Q_esmb as Agent
    agent = Agent(state_dim=1, action_dim=1, **hyperparams)
    
    # Load pre-trained model
    try:
        agent.load_model("Results/mdp", "best")
        print("✓ Loaded pre-trained model")
    except:
        print("⚠ No pre-trained model found. Using untrained agent for demonstration.")
    
    print(f"Expert policy: {policy_mdp}")
    
    # 1. Traditional discrete grid visualization
    print("\n" + "="*50)
    print("1. TRADITIONAL DISCRETE GRID VISUALIZATION")
    print("="*50)
    
    _, frequency_DQ = extract_deterministic_policy(
        agent, N_s, N_a, 
        sample_method="EAS", 
        N_sample=10, 
        N_sampling=100, 
        eta=0.0
    )
    visualize_policy_heatmap(
        frequency_DQ, policy_mdp, 
        title="Traditional Discrete Policy Heat Map", 
        save_path="traditional_discrete_heatmap.png", 
        figsize=(8, 6)
    )
    
    # 2. Continuous policy visualization with interpolation
    print("\n" + "="*50)
    print("2. CONTINUOUS POLICY VISUALIZATION WITH INTERPOLATION")
    print("="*50)
    
    visualize_continuous_policy(
        agent=agent,
        state_range=(-1.0, 1.0),
        action_range=(-1.0, 1.0),
        n_state_points=100,
        n_action_points=100,
        n_samples_per_state=50,
        sample_method="EAS",
        eta=0.1,
        title="Continuous Policy with Interpolation",
        save_path="continuous_policy_interpolation.png",
        figsize=(15, 6),
        interpolation_method="linear"
    )
    
    # 3. 3D continuous policy visualization
    print("\n" + "="*50)
    print("3. 3D CONTINUOUS POLICY VISUALIZATION")
    print("="*50)
    
    visualize_continuous_policy_3d(
        agent=agent,
        state_range=(-1.0, 1.0),
        action_range=(-1.0, 1.0),
        n_state_points=50,
        n_action_points=50,
        n_samples_per_state=30,
        sample_method="EAS",
        eta=0.1,
        title="3D Continuous Policy Visualization",
        save_path="3d_continuous_policy.png",
        figsize=(12, 8)
    )
    
    # 4. Policy comparison: learned vs expert
    print("\n" + "="*50)
    print("4. POLICY COMPARISON: LEARNED VS EXPERT")
    print("="*50)
    
    visualize_policy_comparison(
        agent=agent,
        expert_policy=policy_mdp,
        state_range=(-1.0, 1.0),
        action_range=(-1.0, 1.0),
        n_state_points=100,
        n_action_points=100,
        n_samples_per_state=50,
        sample_method="EAS",
        eta=0.1,
        title="Learned vs Expert Policy Comparison",
        save_path="policy_comparison.png",
        figsize=(18, 6)
    )
    
    # 5. Performance evaluation
    print("\n" + "="*50)
    print("5. PERFORMANCE EVALUATION")
    print("="*50)
    
    eta = 0.1
    sample_method = "EAS"
    
    reward_DQ, _ = evaluate_diffusionQ(
        agent, R, P, N_s, N_a, 
        N_iter=100, 
        sample_method=sample_method, 
        N_sample=10, 
        eta=eta
    )
    
    print(f"Expert's Reward: {np.mean(dataset['rewards']):.4f}")
    print(f"Learned Policy Reward: {reward_DQ:.4f}")
    print(f"Performance Ratio: {reward_DQ / np.mean(dataset['rewards']):.4f}")
    
    # 6. Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("✓ Traditional discrete grid visualization completed")
    print("✓ Continuous policy visualization with interpolation completed")
    print("✓ 3D continuous policy visualization completed")
    print("✓ Policy comparison (learned vs expert) completed")
    print("✓ Performance evaluation completed")
    print("\nGenerated files:")
    print("- traditional_discrete_heatmap.png")
    print("- continuous_policy_interpolation.png")
    print("- 3d_continuous_policy.png")
    print("- policy_comparison.png")
    
    print("\n" + "="*50)
    print("ADVANTAGES OF CONTINUOUS VISUALIZATION")
    print("="*50)
    print("1. Smooth Representation: Instead of discrete grid cells, smooth interpolated surfaces")
    print("2. Better Resolution: Can visualize fine-grained policy behavior")
    print("3. Multiple Views: Heatmap, contour plot, and 3D surface views")
    print("4. Comparison Capabilities: Easy comparison between learned and expert policies")
    print("5. Flexible Interpolation: Different interpolation methods for different use cases")
    print("\nThe continuous visualization provides a much richer understanding of the")
    print("policy's behavior across the entire state-action space.")

if __name__ == "__main__":
    main() 