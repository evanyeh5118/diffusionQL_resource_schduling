# Diffusion-based Resource Scheduling with Offline RL

This project implements **DiffusionQL**, an offline reinforcement learning framework that uses diffusion-based policies for wireless resource scheduling. A hierarchical multi-agent architecture coordinates sub-agents (trained via MDP value iteration) under a central diffusion policy to allocate bandwidth, modulation, and coding resources across users while minimizing packet loss.

Three training paradigms are supported:
- **DiffusionQL** — diffusion policy with ensemble Q-critics (main method)
- **IQL / CQL** — implicit Q-learning and conservative Q-learning via the d3rlpy library
- **Online DRL** — SAC / TD3 baselines via Stable-Baselines3

## Project Structure

```
diffusionQL_resource_schduling/
├── src/
│   ├── difsched/                  # Main package
│   │   ├── agents/
│   │   │   ├── DiffusionQL/       # Diffusion policy + ensemble critics
│   │   │   ├── drl/               # Online RL (SAC, TD3)
│   │   │   ├── dr3rlpy/           # d3rlpy integration (IQL, CQL, BC)
│   │   │   ├── mdp/               # MDP sub-agent solver (value iteration)
│   │   │   └── gym_env/           # Gym environment wrappers
│   │   ├── config/                # All experiment / env / agent configs
│   │   ├── env/                   # Hybrid and SPS environments
│   │   ├── training/              # Training loops
│   │   ├── evaluation/            # Evaluation utilities
│   │   └── utils/                 # EnvInterface, DataSampler, Visualization
│   └── notebooks/                 # Step-by-step workflow (see below)
├── data/
│   ├── processed/
│   │   ├── offline_dataset/       # Generated offline RL datasets
│   │   ├── oai/                   # Raw OAI traffic CSVs
│   │   └── traffic/               # Converted traffic pickle files
│   └── results/
│       ├── dql/                   # Trained DiffusionQL models
│       ├── drl/                   # Trained DRL models
│       └── MdpPolicy/            # Trained MDP sub-agent policies
└── scripts/                       # Standalone example scripts
```

## Notebook Workflow

All notebooks live under `src/notebooks/` and are numbered by stage. Run them **from the repository root** so that `src.difsched` imports resolve correctly.

### Stage 0 — Convert OAI Dataset (optional)

| Notebook | Purpose |
|----------|---------|
| `00_convert_oai_dataset/main_read_and_convert_oai_dataset.ipynb` | Converts raw OAI CSV files in `data/processed/oai/` to pickle format under `data/processed/traffic/` |

Skip this step if the traffic pickle files are already present.

### Stage 1 — Offline Data Pipeline

Run these in order:

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_offlinedata_pipeline/main_mdp_build_agent.ipynb` | Build MDP sub-agents via value iteration (`agent_config` indices 0, 1) and save policies to `data/results/MdpPolicy/` |
| 2 | `01_offlinedata_pipeline/main_gen_offline_dataset.ipynb` | Roll out the MDP sub-agents in the environment to generate offline datasets (`dataset_config` indices 0–7), saved to `data/processed/offline_dataset/` |
| 3 | `01_offlinedata_pipeline/main_mdp_evaluate_agent.ipynb` | (Optional) Evaluate MDP sub-agent performance |

### Stage 2 — Training

Pick one (or more) training notebook depending on the algorithm:

#### DiffusionQL

| Notebook | Purpose |
|----------|---------|
| `02_training/dql_training/long_training.ipynb` | Single long DiffusionQL training run. Set `hyperparams` (diffusion steps, learning rate, number of critics, etc.) and `trainingConfig` (iterations, batch size, BC loss schedule). Uses `exp_config` to select environment and dataset. |
| `02_training/dql_training/mutiple_training.ipynb` | Loop over multiple `exp_config` indices to train several DiffusionQL agents in sequence. Supports both `offline` and `hybrid` training modes. |

Key hyperparameters to configure in the notebook cells:
- `N_diffusion_steps` — number of denoising steps (e.g. 20–30)
- `num_critics` — ensemble size for LCB Q-estimation (e.g. 8–12)
- `lr` — learning rate (e.g. 5e-4)
- `iterations` / `len_period` — total training iterations and steps per period
- `training_type` — `"offline"` (dataset only) or `"hybrid"` (dataset + online rollouts)

#### IQL / CQL (d3rlpy)

| Notebook | Purpose |
|----------|---------|
| `02_training/iql_training/main_prepare_offline_dataset.ipynb` | Convert the offline dataset into d3rlpy-compatible format |
| `02_training/iql_training/main_dr3rlpy_train.ipynb` | Train IQL, CQL, or BC models using d3rlpy |

#### Online DRL (Stable-Baselines3)

| Notebook | Purpose |
|----------|---------|
| `02_training/drl_training/main_drl_train.ipynb` | Train SAC / TD3 agents with early stopping and parallel environments |
| `02_training/drl_training/main_drl_test.ipynb` | Evaluate trained DRL agents |

### Stage 3 — Evaluation

| Notebook | Purpose |
|----------|---------|
| `03_evaluation/long_evaluation.ipynb` | Evaluate a single trained model over long episodes |
| `03_evaluation/multiple_evaluation.ipynb` | Batch-evaluate multiple trained models |
| `03_evaluation/d3rlpy_evaluation.ipynb` | Evaluate d3rlpy-trained models |

### Stage 4 — Analysis & Figures

| Notebook | Purpose |
|----------|---------|
| `04_analysis/fig01_training_reward_compare.ipynb` | Compare training reward curves across configurations |
| `04_analysis/fig02_mix_gaussian_example.ipynb` | Visualize mixture Gaussian diffusion example |
| `04_analysis/fig03_encode_action.ipynb` | Visualize action encoding/decoding |
| `04_analysis/fig04_policy_distribution.ipynb` | Analyze learned policy distributions |
| `04_analysis/figure05_model_accuracy.ipynb` | Model accuracy analysis |

### Stage 5 — Standalone Examples

| Notebook | Purpose |
|----------|---------|
| `05_examples/mdp/main_mdp_learning.ipynb` | MDP value iteration walkthrough |
| `05_examples/mixture_gaussian/main_toy_mixture_gaussion.ipynb` | Toy diffusion model on mixture of Gaussians |
| `05_examples/wireless_model/varying_channel_sim.ipynb` | Wireless channel simulation |

## Configuration System

Four Python config files in `src/difsched/config/` control all experiments:

```
exp_config  ──→  env_config   ──→  evaluation environment
            ──→  dataset_config ──→  offline dataset generation
            ──→  dataset_config ──→  agent_config ──→  sub-agent MDP training
```

| File | Role |
|------|------|
| `exp_config.py` | Top-level experiment orchestrator: selects which dataset, environment, and agent configs to use |
| `simenv_configs.py` | Environment parameters: user count, bandwidth, traffic pattern, sigmoid reward params |
| `agent_config.py` | Sub-agent MDP configs: smaller-scale envs for individual agent training |
| `dataset_config.py` | Offline dataset generation configs: environment variations to create diverse training data |

### Dataset Config Index Mapping

| Index | Users | Dataflow | Sub-agents | Usage |
|-------|-------|----------|------------|-------|
| 0 | 8 | thumb_fr | [0,0] | Training (simple) |
| 1 | 20 | thumb_fr | [0,0,0,0,0] | Training (complex) |
| 2 | 8 | thumb_fr | [0,0] | Testing |
| 3 | 20 | thumb_fr | [0,0,0,0,0] | Testing |
| 4 | 8 | thumb_bk | [1,1] | Training (simple) |
| 5 | 20 | thumb_bk | [1,1,1,1,1] | Training (complex) |
| 6 | 8 | thumb_bk | [1,1] | Testing |
| 7 | 20 | thumb_bk | [1,1,1,1,1] | Testing |

Sub-agent indices 0 and 1 refer to `agent_config.py` entries (thumb_fr and thumb_bk traffic patterns respectively).

## Requirements

- Python 3.9+
- PyTorch (with CUDA recommended)
- d3rlpy (for IQL/CQL training)
- stable-baselines3 (for online DRL baselines)
- numpy, matplotlib, pickle

## License

See [LICENSE](LICENSE).
