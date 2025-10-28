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

def loadAndEvaluation(env, envInterface, dataset_expert, modelFolder, exp_idx_list=[0]):
    rewards_expert = dataset_expert['rewardRecord']
    print(f"Expert's Reward: {np.mean(rewards_expert)}")
    #=============================================
    #================ Best Model ================
    #=========================================
    best_reward = np.inf
    best_model_idx = None
    agent_list = []
    for exp_idx in exp_idx_list:
        with open(f"{modelFolder}/hyperparams_{exp_idx}.pkl", "rb") as f:
            hyperparams = pickle.load(f)
        agent = Agent(
            state_dim=envInterface.state_dim, 
            action_dim=envInterface.action_dim, 
            **hyperparams
        )
        agent.load_model(modelFolder, f'{exp_idx}_best')
        agent_list.append(agent)
        env.reset()
        env.selectMode(mode="test", type="data")
        reward, _ = eval(
            agent, env, envInterface, 
            LEN_eval=250, obvMode="predicted", sample_method="greedy", 
            N_action_candidates=50, eta=0.1, verbose=True) 
        print(f"reward_diffusionQ{exp_idx}: {reward}")
        if reward < best_reward:
            best_reward = np.mean(reward)
            best_model_idx = exp_idx

    print(f"best_exp_idx: {best_model_idx}")
    agent = agent_list[best_model_idx]

    #=========================================
    #================ Evaluation ================
    #=========================================
    env.selectMode(mode="test", type="data")

    LEN_eval = 50
    reward_expert_list = []
    reward_dql_low_eta_list = []
    reward_dql_high_eta_list = []
    for _ in tqdm(range(20)):
        env.reset()
        env.selectMode(mode="test", type="data")
        reward_expert_sample = np.random.choice(rewards_expert, size=LEN_eval, replace=False)
        reward_dql_low_eta, _ = eval(agent, env, envInterface, LEN_eval=LEN_eval, obvMode="predicted", 
                                sample_method="greedy", N_action_candidates=50, eta=0.01, verbose=True) 
        reward_dql_high_eta, _ = eval(agent, env, envInterface, LEN_eval=LEN_eval, obvMode="predicted", 
                                sample_method="greedy", N_action_candidates=50, eta=1.0, verbose=True) 
        reward_expert_list.append(np.mean(reward_expert_sample))
        reward_dql_low_eta_list.append(reward_dql_low_eta)
        reward_dql_high_eta_list.append(reward_dql_high_eta)

    reward_expert_list = np.array(reward_expert_list)
    reward_dql_low_eta_list = np.array(reward_dql_low_eta_list)
    reward_dql_high_eta_list = np.array(reward_dql_high_eta_list)

    mean_exp, bound_exp = mean_confidence_interval(reward_expert_list)
    mean_dql_low_eta, bound_dql_low_eta = mean_confidence_interval(reward_dql_low_eta_list)
    mean_dql_high_eta, bound_dql_high_eta = mean_confidence_interval(reward_dql_high_eta_list)

    print(f"Expert Reward: {mean_exp:.6f} ± {bound_exp:.6f}")
    print(f"DQL Reward (low eta): {mean_dql_low_eta:.6f} ± {bound_dql_low_eta:.6f}")
    print(f"DQL Reward (high eta): {mean_dql_high_eta:.6f} ± {bound_dql_high_eta:.6f}")
