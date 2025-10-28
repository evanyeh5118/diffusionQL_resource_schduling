
import sys
sys.path.insert(0, '../../../../')

import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.difsched.config import getExpConfig, visualizeExpConfig
from src.difsched.config import getEnvConfig, visualizeEnvConfig, getDatasetConfig, visualizeDatasetConfig
from src.difsched.utils.DataSampler import ReplayBuffer, ReplayBufferHybrid
from src.difsched.utils.Visualization import MultiLivePlot
from src.difsched.env.EnvironmentSim import createEnv
from src.difsched.utils.EnvInterface import EnvInterface
from src.difsched.evaluation import eval
from src.difsched.agents.DiffusionQL.DQL_Q_esmb import DQL_Q_esmb as Agent

def training(trainingConfig, dataset_off, hyperparams, env, envInterface, save_folder, N_exp_list=[0,1,2]):
    BC_loss = trainingConfig.get('BC_loss', True)
    iterations = trainingConfig.get('iterations', 100)
    batch_size = trainingConfig.get('batch_size', 100)
    LEN_eval = trainingConfig.get('LEN_eval', 50)
    report_period = trainingConfig.get('report_period', 10)
    len_period = trainingConfig.get('len_period', 50)
    warm_up_period = trainingConfig.get('warm_up_period', len_period)
    max_sp_ratio = trainingConfig.get('max_sp_ratio', 1.0)
    min_sp_ratio = trainingConfig.get('min_sp_ratio', 0.5)
    max_weight_bc_loss = trainingConfig.get('max_weight_bc_loss', 1.0)
    min_weight_bc_loss = trainingConfig.get('min_weight_bc_loss', 1.0)
    rb_capacity = trainingConfig.get('rb_capacity', 30000)

    print(max_weight_bc_loss, min_weight_bc_loss)

    for N_exp in N_exp_list:
        with open(f"{save_folder}/hyperparams_{N_exp}.pkl", "wb") as f:
            pickle.dump(hyperparams, f)
 
        dataSamplerOff = ReplayBuffer(capacity=rb_capacity, envInterface=envInterface, device=hyperparams['device'])
        dataSamplerOn = ReplayBufferHybrid(capacity=rb_capacity, envInterface=envInterface, device=hyperparams['device'])
        dataSamplerOff.add(dataset_off)
        if BC_loss == True:
            dataSamplerOn.addOffline(dataset_off)
        batch = dataSamplerOff.sample(len(dataSamplerOff))
        print(f"Expert's Reward: {np.mean(batch[2].cpu().detach().numpy())}")

        print(f"state_dim: {envInterface.state_dim}, action_dim: {envInterface.action_dim}")
        agent = Agent(
            state_dim=envInterface.state_dim, 
            action_dim=envInterface.action_dim, 
            **hyperparams
        )
        metrics_train = {'Ld': [], 'Lq': [], 'Le': [], 'loss_Q': [], 'Reward': []}
        ploter = MultiLivePlot(nrows=1, ncols=5, titles=["Ld", "Lq", "Le", "loss_Q", "Reward"], display_window=25)
        best_reward = np.inf
        idx_episode = 1
        while(True):
            if BC_loss == True:
                metrics = agent.train_split(dataSamplerOff, dataSamplerOn, iterations, batch_size, tqdm_pos=0)
            else:
                metrics = agent.train(dataSamplerOn, iterations, batch_size, tqdm_pos=0)
            _, explore_data = eval(agent, env, envInterface, LEN_eval=LEN_eval, obvMode="predicted", 
                                sample_method="exploration", N_action_candidates=10, 
                                eta=np.random.uniform(0.5, 3.0), verbose=True)
            dataSamplerOn.addOnline(explore_data)
            reward, offpolicy_data = eval(agent, env, envInterface, LEN_eval=LEN_eval, obvMode="predicted", 
                                        sample_method="greedy", N_action_candidates=50, eta=1.0, verbose=True)
            dataSamplerOn.addOnline(offpolicy_data)
            sample_ratio = np.max([min_sp_ratio, max_sp_ratio - ((max_sp_ratio-min_sp_ratio)/warm_up_period) * idx_episode])
            dataSamplerOn.set_sample_ratio(sample_ratio)
            weight_bc_loss = np.max([min_weight_bc_loss, max_weight_bc_loss - ((max_weight_bc_loss-min_weight_bc_loss)/warm_up_period) * idx_episode])
            agent.set_weight_bc_loss(weight_bc_loss)

            metrics_train['Ld'] += metrics['Ld']
            metrics_train['Lq'] += metrics['Lq']
            metrics_train['Le'] += metrics['Le']
            metrics_train['loss_Q'] += metrics['loss_Q']
            metrics_train['Reward'].append(reward)
            ploter.update(0, idx_episode, np.mean(metrics['Ld']))
            ploter.update(1, idx_episode, np.mean(metrics['Lq']))
            ploter.update(2, idx_episode, np.mean(metrics['Le']))
            ploter.update(3, idx_episode, np.mean(metrics['loss_Q']))
            ploter.update(4, idx_episode, reward)    
        
            if idx_episode > 10:
                window = 5
                smooth_reward = np.convolve(
                    np.concatenate([np.zeros(window), np.array(metrics_train['Reward'])]), 
                    np.ones(window)/window, mode='valid')[1:]
                # save model
                if smooth_reward[-1] < best_reward:
                    best_reward = smooth_reward[-1]
                    agent.save_model(save_folder, f'{N_exp}_best')
                    print(f"save model {N_exp}_best, smoothed reward: {best_reward}")
                    with open(f"{save_folder}/train_metrics_{N_exp}_best.pkl", "wb") as f:
                        pickle.dump(metrics_train, f)
                # stop training
                if np.abs(smooth_reward[-1] - smooth_reward[-2]) < 1e-6 or \
                    idx_episode > len_period:
                    #smooth_reward[-1] > 5.0*smooth_reward[-window] or \
                    agent.save_model(save_folder, f'{N_exp}_end')
                    with open(f"{save_folder}/train_metrics_{N_exp}_end.pkl", "wb") as f:
                        pickle.dump(metrics_train, f)
                    break

            if idx_episode % report_period == 0:
                print("=" * 20 + f"Iteration {idx_episode}" + "=" * 20)
                print(f"Ld: {np.mean(metrics['Ld'])}, " + 
                    f"Lq: {np.mean(metrics['Lq'])}, " + 
                    f"Le: {np.mean(metrics['Le'])}, " + 
                    f"loss_Q: {np.mean(metrics['loss_Q'])}")
                print(f"Avg. Reward: {np.mean(metrics_train['Reward'][-int(report_period):])}, sample_ratio: {sample_ratio}, weight_bc_loss: {weight_bc_loss}")
                print("=" * 50) 
            idx_episode += 1