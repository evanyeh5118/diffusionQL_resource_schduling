import numpy as np
import torch
from tqdm import tqdm


def _observation_helper(
        env,
        obvMode: str = "perfect", 
    ):
    u, u_predicted = env.getStates()
    if obvMode == "perfect":
        return u
    elif obvMode == "predicted":
        return u_predicted
    else:
        raise ValueError(f"Invalid observation mode: {obvMode}")

def _step(
        agent,
        env, 
        envInterface,
        obvMode: str = "perfect", 
        sample_method: str = "greedy", 
        N_action_candidates: int = 10,
        eta: float = 0.01,
    ):
    u = _observation_helper(env, obvMode)    
    s = envInterface.preprocess_state(u)
    s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0).to(agent.device)
    if sample_method == "exploration":
        a = agent.sample(
            s, 
            sample_method="EAS",
            N=N_action_candidates, 
            eta = eta
        ).cpu().detach().numpy()[0]
    else:
        a = agent.sample(
            s, 
            sample_method=sample_method,
            N=N_action_candidates, 
            eta = eta,
        ).cpu().detach().numpy()[0]
    r = envInterface.postprocess_action(a)
    reward = env.applyActions(r)
    u_next = _observation_helper(env, obvMode)
    env.updateStates()
    return u, r, reward, u_next

def eval(
        agent,
        env,
        envInterface, 
        LEN_eval=1000, 
        obvMode="perfect", 
        mode="test",
        type="data",
        sample_method="greedy",
        N_action_candidates=10,
        eta=0.01,
        verbose=True
    ):
    env.reset()
    env.selectMode(mode=mode, type=type)
    info = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}
    for window in tqdm(range(LEN_eval), desc="Evaluation windows", leave=False, disable=not verbose):
        u, action, reward, u_next = _step(
            agent, env, envInterface, obvMode, sample_method, N_action_candidates, eta
        )
        #============ Record Results ============
        info['observations'].append(u)
        info['actions'].append(action)
        info['rewards'].append(reward)
        info['next_observations'].append(u_next)
    reward = np.mean(info['rewards'])
    return reward, info