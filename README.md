# Diffusion-based Resource Scheduling with Offline RL

A minimal codebase for scheduling using offline reinforcement learning with diffusion-based policies.

## Configuration Structure

The configuration system is organized hierarchically with four main configuration files:

1. **`exp_config.py`** (Experiment Config) - Top-level configuration that references other configs:
   - Links dataset configs for training and testing
   - Specifies which environment config to use
   - Acts as the experiment orchestrator

2. **`simenv_configs.py`** (Simulation Environment Config) - Defines the evaluation environment:
   - Network parameters (N_user, B, r_bar)
   - Traffic patterns (dataflow, window settings)
   - Multi-agent setup (sub_agents_idx, user_map)
   - Sigmoid parameters for reward functions

3. **`agent_config.py`** (Sub-Agent Config) - Configuration for individual sub-agents:
   - Smaller-scale environment for sub-agent training
   - Reduced user count and bandwidth allocation
   - Individual agent resource constraints
   - Used to train MDP-based sub-policies

4. **`dataset_config.py`** (Dataset Config) - Configuration for generating offline datasets:
   - Similar to env_config but with variations
   - Different sigmoid parameters to create diverse training data
   - Used to generate expert demonstrations
   - Supports both training and testing dataset configurations

**Configuration Flow:**
```
exp_config → env_config → evaluation environment
exp_config → dataset_config → offline dataset generation
exp_config → dataset_config → agent_config → sub-agent training
```

The `exp_config` ties everything together by selecting specific indices for dataset, agent, and environment configurations for each experiment.

## Using Offline Data Pipeline Notebooks

The notebooks in `notebooks/01_offlinedata_pipeline/` follow a specific sequence with config index assignments:

### Pipeline Workflow

1. **`main_mdp_build_agent.ipynb`** - Build MDP sub-agents:
   - Uses `agent_config.py` indices (0, 1)
   - Config index 0 → thumb_fr dataflow, 4 users
   - Config index 1 → thumb_bk dataflow, 4 users
   - Saves trained MDP policies to `data/results/MdpPolicy/mdp_config{idx}.pkl`

2. **`main_gen_offline_dataset.ipynb`** - Generate offline datasets:
   - Uses `dataset_config.py` indices (0-7)
   - For each dataset config index:
     - Loads corresponding environment settings
     - Retrieves sub-agent policies using `sub_agents_idx` from dataset config
     - Runs simulations to generate training data
     - Saves to `data/processed/offline_dataset/subOptimalAgent_encConfig{idx}.pkl`

3. **`main_mdp_evaluate_agent.ipynb`** - Evaluate MDP agents:
   - Uses `agent_config.py` indices to load trained policies
   - Runs evaluation on test environment
   - Visualizes performance metrics

### Config Index Mapping

| Dataset Config | Users | Dataflow | Sub-agents | Usage |
|---------------|-------|----------|------------|-------|
| 0 | 8 | thumb_fr | [0,0] | Training (simple) |
| 1 | 20 | thumb_fr | [0,0,0,0,0] | Training (complex) |
| 2 | 8 | thumb_fr | [0,0] | Testing (expanded sigmoid) |
| 3 | 20 | thumb_fr | [0,0,0,0,0] | Testing (expanded sigmoid) |
| 4 | 8 | thumb_bk | [1,1] | Training (simple) |
| 5 | 20 | thumb_bk | [1,1,1,1,1] | Training (complex) |
| 6 | 8 | thumb_bk | [1,1] | Testing (expanded sigmoid) |
| 7 | 20 | thumb_bk | [1,1,1,1,1] | Testing (expanded sigmoid) |

**Key Points:**
- `sub_agents_idx` from dataset config determines which sub-agent policies to load
- Sub-agent indices 0 and 1 refer to `agent_config.py` indices
- Each dataset config generates one offline dataset file
- The workflow is: Build agents (config 0,1) → Generate datasets (config 0-7) → Evaluate

```
diffusion_resource_schduling_intra_slice/
├── README.md
├── LICENSE
├── docs/
│   ├── DELIVERY_SUMMARY.txt
│   ├── EXECUTIVE_SUMMARY.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── PROJECT_STRUCTURE_ANALYSIS.md
│   └── PROJECT_STRUCTURE_COMPARISON.md
├── data/
│   ├── processed/
│   │   ├── offline_dataset/
│   │   └── traffic/
│   └── results/
│       ├── dql/
│       ├── drl/
│       ├── examples/
│       └── MdpPolicy/
├── src/
│   └── difsched/
│       ├── agents/
│       │   ├── DiffusionQL/
│       │   ├── drl/
│       │   └── mdp/
│       ├── config/
│       ├── data/
│       ├── env/
│       ├── evaluation/
│       ├── training/
│       └── utils/
├── notebooks/
│   ├── 00_getting_started/
│   ├── 01_offlinedata_pipeline/
│   ├── 02_training/
│   ├── 03_evaluation/
│   ├── 04_analysis/
│   └── 05_examples/
└── scripts/
    └── examples/
        ├── mdp/
        └── mixture_gaussian/
```
