import numpy as np
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
from tqdm import tqdm

from src.difsched.agents.DiffusionQL.DQL_Q_esmb import DQL_Q_esmb as Agent
from src.difsched.evaluation import eval

def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = data.size
    m = np.mean(data)
    se = stats.sem(data, axis=None)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def loadAndEvaluation(env, envInterface, dataset_expert, modelFolder, exp_idx_list=[0], eta=1.0, N_action_candidates=100):
    rewards_expert = dataset_expert['rewardRecord']
    print(f"Expert's Reward: {np.mean(rewards_expert)}")
    #=============================================
    #================ Best Model ================
    #=========================================
    best_reward = np.inf
    best_model_idx = None
    agent_list = []
    all_rewards = []
    
    for exp_idx in exp_idx_list:
        with open(f"{modelFolder}/hyperparams_{exp_idx}.pkl", "rb") as f:
            hyperparams = pickle.load(f)
        agent = Agent(
            state_dim=envInterface.state_dim, 
            action_dim=envInterface.action_dim, 
            **hyperparams
        )
        print(f"Loading model {exp_idx}_best")
        agent.load_model(modelFolder, f'{exp_idx}_best')
        agent_list.append(agent)
        
        env.reset()
        env.selectMode(mode="test", type="data")
        _, info = eval(
            agent, env, envInterface, 
            LEN_eval=2500, obvMode="predicted", sample_method="greedy", 
            N_action_candidates=N_action_candidates, eta=eta, verbose=True) 
        
        reward_mean = np.mean(info['rewards'])
        reward_std = np.std(info['rewards'])
        print(f"reward_diffusionQ{exp_idx}: mean={reward_mean:.4f}, std={reward_std:.4f}")
        


