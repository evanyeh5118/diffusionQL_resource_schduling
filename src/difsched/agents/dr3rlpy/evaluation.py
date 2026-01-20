
import numpy as np

# Prefer Gymnasium; fallback to Gym.
import gymnasium as gym



# -----------------------------
# 2) API-compat helpers (Gym vs Gymnasium)
# -----------------------------
def reset_compat(env: gym.Env) -> np.ndarray:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
    else:
        obs = out
    return np.asarray(obs, dtype=np.float32)


def step_compat(env: gym.Env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        terminated, truncated = bool(done), False
    return (
        np.asarray(obs, dtype=np.float32),
        float(reward),
        done,
        terminated,
        truncated,
        info,
    )


# -----------------------------
# 5) Evaluate a learned policy in the env
# -----------------------------
def evaluate(algo, env: gym.Env, n_episodes: int = 5) -> float:
    returns = []
    for _ in range(n_episodes):
        obs = reset_compat(env)
        done = False
        ep_ret = 0.0
        step_count = 0
        while not done:
            # d3rlpy expects batched observations for predict(). :contentReference[oaicite:5]{index=5}
            act = algo.predict(obs[None, ...])[0]
            if isinstance(env.action_space, gym.spaces.Discrete):
                act = int(act)
            obs, r, done, *_ = step_compat(env, act)
            ep_ret += r
            step_count += 1
        avg_reward = ep_ret / step_count if step_count > 0 else 0.0
        returns.append(avg_reward)
    return float(np.mean(returns))