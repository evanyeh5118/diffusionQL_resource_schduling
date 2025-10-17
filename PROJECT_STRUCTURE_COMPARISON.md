# Project Structure Comparison: Before vs. After

## Visual Structure Comparison

### BEFORE: Current Scattered Structure ❌

```
diffusion_resource_schduling_intra_slice/
├── Agents/                          ← Multiple agent types scattered
│   ├── Agents/MdpPolicy/            ← Confusing nested "Agents"
│   ├── DQL/
│   ├── DrlLibs/
│   └── ModelBasedSolvers/
├── Configs/                         ← Configuration loose structure
├── Datasets/                        ← Mixed: code + data + notebooks
│   ├── DatasetManagerLibs/
│   ├── OfflineDataset/
│   └── TrafficDataset/
├── DiffusionQL/                     ← Separate from Agents/
├── Environment/                     ← Domain specific utils
│   ├── EnvironmentSim.py
│   └── Helpers/
├── Examples/                        ← Scattered research artifacts
├── Figures/                         ← More research artifacts
├── Helpers/                         ← Another helpers directory
├── demo00_*.ipynb (8 files)         ← Unorganized notebooks at root
├── ❌ No requirements.txt
├── ❌ No setup.py
├── ❌ Minimal README
├── ❌ No tests/
└── src/difsched/                    ← Incomplete, unused structure
    ├── agents/
    └── env/
```

**Problems:**
- 🔴 No single entry point
- 🔴 Unclear module boundaries
- 🔴 Mixed concerns everywhere
- 🔴 Not installable as package
- 🔴 Difficult to test
- 🔴 Scattered notebooks

---

### AFTER: Proposed Clean Structure ✅

```
diffusion-offrl/
├── 📄 README.md                     ← Comprehensive documentation
├── 📄 requirements.txt              ← Explicit dependencies
├── 📄 setup.py                      ← Package installation
├── 📄 .gitignore                    ← Version control
├── 📄 LICENSE                       ← Legal compliance
│
├── src/difsched/                    ← ✅ Clean, installable package
│   ├── __init__.py
│   ├── config/                      ← Centralized configuration
│   │   ├── base_config.py
│   │   ├── env_configs.py
│   │   ├── model_configs.py
│   │   └── schemas.py
│   │
│   ├── env/                         ← Environment module
│   │   ├── environment.py
│   │   ├── traffic.py
│   │   ├── simulators.py
│   │   ├── models.py
│   │   └── rewards.py
│   │
│   ├── agents/                      ← Unified agents module
│   │   ├── base.py
│   │   ├── dql/
│   │   │   ├── agent.py
│   │   │   ├── actors.py
│   │   │   ├── critics.py
│   │   │   └── variants.py
│   │   ├── drl/
│   │   │   ├── agent.py
│   │   │   └── variants.py
│   │   └── mdp/
│   │       ├── mdp_builder.py
│   │       ├── mdp_solver.py
│   │       └── policies.py
│   │
│   ├── training/                    ← Training pipelines
│   │   ├── offline_training.py
│   │   ├── online_training.py
│   │   ├── callbacks.py
│   │   └── utils.py
│   │
│   ├── evaluation/                  ← Evaluation & analysis
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   ├── data/                        ← Data management code
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   ├── processors.py
│   │   ├── generators.py
│   │   └── sampler.py
│   │
│   └── utils/                       ← Consolidated utilities
│       ├── logging.py
│       ├── checkpointing.py
│       └── helpers.py
│
├── tests/                           ← ✅ Test structure
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── notebooks/                       ← ✅ Organized notebooks
│   ├── 00_getting_started/
│   ├── 01_data_pipeline/
│   ├── 02_training/
│   ├── 03_evaluation/
│   ├── 04_analysis/
│   └── 05_benchmarks/
│
├── configs/                         ← ✅ Configuration files
│   ├── env/
│   ├── models/
│   └── training/
│
├── data/                            ← ✅ Data storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── scripts/                         ← ✅ CLI utilities
│   ├── train_agent.py
│   ├── evaluate_agent.py
│   └── generate_dataset.py
│
├── docs/                            ← ✅ Documentation
│   ├── ARCHITECTURE.md
│   ├── GETTING_STARTED.md
│   └── API.md
│
└── .github/
    └── workflows/                   ← ✅ CI/CD pipelines
```

**Benefits:**
- 🟢 Clear module hierarchy
- 🟢 Single package (`difsched`)
- 🟢 Installable via `pip install -e .`
- 🟢 Organized by feature, not by type
- 🟢 Easy to locate code
- 🟢 Testable structure
- 🟢 Production-ready

---

## Module Mapping: Migration Guide

### Agents Migration

| Current Location | New Location | Note |
|---|---|---|
| `Agents/DQL/dql.py` | `src/difsched/agents/dql/agent.py` | Main DQL agent |
| `Agents/DQL/main_test.ipynb` | `notebooks/03_evaluation/` | Moved to evaluation |
| `DiffusionQL/Actors.py` | `src/difsched/agents/dql/actors.py` | DQL actors |
| `DiffusionQL/Critics.py` | `src/difsched/agents/dql/critics.py` | DQL critics |
| `DiffusionQL/DQL_Q_iql.py` | `src/difsched/agents/dql/variants.py` | DQL variant |
| `Agents/DrlLibs/` | `src/difsched/agents/drl/` + `src/difsched/training/` | Split: agent code + training |
| `Agents/ModelBasedSolvers/` | `src/difsched/agents/mdp/` | MDP solver agents |

### Environment Migration

| Current Location | New Location | Note |
|---|---|---|
| `Environment/EnvironmentSim.py` | `src/difsched/env/environment.py` | Main environment |
| `Environment/RewardFuntions.py` | `src/difsched/env/rewards.py` | Reward functions |
| `Environment/Helpers/Simulators.py` | `src/difsched/env/simulators.py` | Wireless simulators |
| `Environment/Helpers/TrafficGenerator.py` | `src/difsched/env/traffic.py` | Traffic generation |
| `Environment/Helpers/WirelessModel.py` | `src/difsched/env/models.py` | Wireless models |
| `Environment/PolicySimulator.py` | `src/difsched/evaluation/` | Policy evaluation |

### Data Migration

| Current Location | New Location | Note |
|---|---|---|
| `Datasets/DatasetManagerLibs/` | `src/difsched/data/` | Data management code |
| `Datasets/OfflineDataset/*.pkl` | `data/processed/offline_dataset/` | Data files (gitignored) |
| `Datasets/TrafficDataset/TrafficData/` | `data/raw/traffic/` | Raw traffic data |
| `Datasets/main_gen_offline_dataset.ipynb` | `notebooks/01_data_pipeline/` | Data pipeline notebook |
| `Datasets/main_gen_traffic_dataset.ipynb` | `notebooks/01_data_pipeline/` | Data generation notebook |

### Configuration Migration

| Current Location | New Location | Note |
|---|---|---|
| `Configs/EnvConfigs.py` | `src/difsched/config/env_configs.py` | Environment config |
| `Configs/DatasetConfigs.py` | `src/difsched/config/` | Dataset config |
| `Configs/PredictorConfigs.py` | `src/difsched/config/` | Predictor config |
| Hardcoded params | `configs/env/default.yaml` | Externalize to YAML |

### Notebooks Migration

| Current Location | New Location | Workflow |
|---|---|---|
| `demo00_valid_env.ipynb` | `notebooks/00_getting_started/02_validate_simulator.ipynb` | Getting started |
| `demo00_interface.ipynb` | `notebooks/00_getting_started/03_interface_demo.ipynb` | Getting started |
| `main_gen_traffic_dataset.ipynb` | `notebooks/01_data_pipeline/01_generate_traffic_dataset.ipynb` | Data pipeline |
| `demo01a_trainOffline_diffusionQ.ipynb` | `notebooks/02_training/01_train_diffusion_ql.ipynb` | Training |
| `demo01c_eval_diffusionQ.ipynb` | `notebooks/03_evaluation/01_eval_diffusion_ql.ipynb` | Evaluation |
| `fig01_training_reward_compare.ipynb` | `notebooks/04_analysis/01_training_reward_comparison.ipynb` | Analysis |
| `demo02_run_all_experiments.ipynb` | `notebooks/05_benchmarks/01_run_all_experiments.ipynb` | Benchmarks |

### Helpers Consolidation

| Current Location | New Location | Note |
|---|---|---|
| `Helpers/DataSampler.py` | `src/difsched/data/sampler.py` | Data sampling |
| `Helpers/EnvInterface.py` | `src/difsched/env/` or `src/difsched/utils/` | Environment interface |
| `Helpers/Eval.py` | `src/difsched/evaluation/evaluator.py` | Evaluation |
| `Helpers/Visualization.py` | `src/difsched/evaluation/visualization.py` | Visualization |
| `DiffusionQL/Helpers.py` | `src/difsched/agents/dql/helpers.py` | DQL helpers |
| `Environment/Helpers/` | `src/difsched/env/` | Environment helpers |

---

## Import Changes Example

### Before (Current)
```python
# In notebook or script
import sys
sys.path.append('../Agents')
sys.path.append('../Environment')
sys.path.append('../Datasets')

from DQL.dql import DQLAgent
from DrlLibs.training import train
from Environment.EnvironmentSim import Environment
from Helpers.Visualization import plot_results
```

### After (Proposed)
```python
# In notebook or script
from difsched.agents.dql import DQLAgent
from difsched.training import offline_training
from difsched.env import Environment
from difsched.evaluation import visualization

# Or using installed package
from difsched import agents, training, env
```

---

## Configuration Changes Example

### Before (Current)
```python
# In EnvConfigs.py
def getEnvConfig(configIdx):
    if configIdx == 0:
        return {
            'N_user': 8,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_fr',
            # ... hardcoded values
        }
```

### After (Proposed)

**configs/env/default.yaml:**
```yaml
environment:
  N_user: 8
  LEN_window: 200
  N_aggregation: 4
  dataflow: thumb_fr
  B: 100
  r_bar: 5
  randomSeed: 999
  sigmoid_k_list: [0.1, 0.2, 0.3, 0.4, 0.5]
  sigmoid_s_list: [10.0, 10.0, 10.0, 10.0, 10.0]
```

**src/difsched/config/base_config.py:**
```python
from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class EnvironmentConfig:
    N_user: int
    LEN_window: int
    N_aggregation: int
    dataflow: str
    B: int
    r_bar: int
    randomSeed: int = 999
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data['environment'])

# Usage
config = EnvironmentConfig.from_yaml('configs/env/default.yaml')
env = Environment(config)
```

---

## Directory Tree Comparison

### Before: Code Locations Scattered

```
Root
 ├─ Main code in 4+ directories
 ├─ Data mixed with code
 ├─ Config mixed with code
 ├─ Utilities in 3+ directories
 ├─ Notebooks scattered at root
 ├─ src/ incomplete and unused
 └─ No clear organization
```

### After: Code Organization Pyramid

```
Root
 ├─ src/difsched/ ← ALL code in one package
 │   ├─ config/
 │   ├─ env/
 │   ├─ agents/
 │   ├─ training/
 │   ├─ evaluation/
 │   ├─ data/
 │   └─ utils/
 ├─ notebooks/ ← Research & demos
 ├─ tests/ ← Quality assurance
 ├─ scripts/ ← CLI tools
 ├─ configs/ ← Configuration files
 ├─ data/ ← Data storage
 └─ docs/ ← Documentation
```

---

## Implementation Summary

### Phase 1: Foundation (Week 1)
- Create new `src/difsched/` structure
- Move core modules (env, basic agents)
- Update imports
- Test basic functionality

### Phase 2: Complete Core (Week 2)
- Migrate all agent types (DQL, DRL, MDP)
- Migrate training modules
- Create configuration system
- Add setup.py and requirements.txt

### Phase 3: Data & Evaluation (Week 3)
- Organize data pipeline
- Migrate evaluation code
- Update notebooks structure
- Add utility modules

### Phase 4: Polish & Documentation (Week 4)
- Write comprehensive README
- Create documentation
- Add tests
- Setup CI/CD

---

## Success Metrics

After restructuring, you should achieve:

✅ **Code Quality**
- Clear module boundaries
- Reduced circular dependencies
- Improved code reusability

✅ **Developer Experience**
- Can install with `pip install -e .`
- Clear import paths
- Easy to understand structure

✅ **Project Maturity**
- Testable code
- Production-ready
- Proper documentation

✅ **Maintenance**
- Easy to locate code
- Quick onboarding for new developers
- Reduced technical debt
