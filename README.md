# Diffusion-based Resource Scheduling with Offline RL

A comprehensive research framework for intelligent resource scheduling in wireless networks using diffusion-based offline reinforcement learning techniques.

**Keywords**: Offline Reinforcement Learning, Diffusion Models, Resource Allocation, Wireless Networks, 5G/6G, Network Slicing

| Aspect | Description |
|--------|-------------|
| **Problem** | Dynamic resource allocation in wireless network slices with time-varying traffic |
| **Approach** | Diffusion Q-Learning (DQL) for learning complex action distributions |
| **Domain** | Next-generation wireless networks (5G/6G) |
| **Key Innovation** | Combining diffusion models with offline RL for policy learning |
| **Status** | Active Research Project |

## 📋 Project Summary

This project implements advanced reinforcement learning methods for **intra-slice resource scheduling** in next-generation wireless networks (5G/6G). The framework addresses the critical challenge of efficiently allocating limited bandwidth resources to multiple users with varying traffic patterns and quality-of-service (QoS) requirements.

### Key Innovation

The project introduces **Diffusion Q-Learning (DQL)**, which leverages diffusion models to learn complex action distributions for resource allocation tasks. This approach is particularly effective for offline reinforcement learning scenarios where interactions with the environment are costly or limited.

### Research Focus

- **Problem**: Dynamic resource allocation in wireless slice management with uncertain traffic patterns
- **Solution**: Diffusion-based policy learning combined with offline RL algorithms
- **Application**: Ultra-reliable low-latency communications (URLLC) and enhanced mobile broadband (eMBB) services

## 🎯 Overview

This project implements advanced RL techniques for intra-slice resource scheduling in next-generation wireless networks. It includes:

- **Agents**: Diffusion Q-Learning (DQL), Deep RL, and MDP-based agents
- **Environment**: Realistic wireless network simulation with traffic generation
- **Training**: Offline reinforcement learning pipeline with behavioral cloning
- **Evaluation**: Comprehensive evaluation and analysis tools
- **Data Pipeline**: Automatic traffic dataset generation and processing

### Core Components

1. **Diffusion Q-Learning (DQL)**: Combines diffusion models with Q-learning for complex action distribution learning
2. **Environment Simulation**: Realistic wireless network with time-varying channel conditions and traffic patterns
3. **Offline RL Pipeline**: Train policies from fixed datasets without online environment interaction
4. **Behavioral Cloning**: Leverage expert demonstrations to bootstrap RL policies
5. **Evaluation Framework**: Comprehensive benchmarks comparing DQL, DRL, and MDP-based approaches

## ✨ Features

- ✅ Professional package structure (`src/difsched/`)
- ✅ Centralized configuration system (YAML-based)
- ✅ Organized Jupyter notebooks by workflow
- ✅ Clear module boundaries and imports
- ✅ Installable as a Python package
- ✅ Comprehensive documentation
- ✅ Data pipeline management

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd diffusion-offrl

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[notebooks]"
```

### Verify Installation

```bash
python -c "from difsched import env, agents, config; print('✓ Installation successful!')"
```

## 🚀 Quick Start

### 1. Setup Environment

```python
from difsched.config import ConfigManager

# Load configuration
cfg_mgr = ConfigManager(config_dir='configs')
env_config = cfg_mgr.load_env_config('default')
print(env_config)
```

### 2. Create Environment

```python
from difsched.env import Environment

env = Environment(env_config.to_dict(), trafficGenerator)
```

### 3. Train Agent

```python
from difsched.training import offline_training

# Train agent (when training module is implemented)
# agent = offline_training(env, config)
```

## 📚 Documentation

### Available Documents

1. **README.md** (this file) - Project overview and quick start
2. **EXECUTIVE_SUMMARY.md** - High-level analysis and recommendations
3. **PROJECT_STRUCTURE_ANALYSIS.md** - Detailed problem analysis
4. **PROJECT_STRUCTURE_COMPARISON.md** - Before/after structure mapping
5. **IMPLEMENTATION_GUIDE.md** - Step-by-step implementation instructions
6. **STRUCTURE_VISUALIZATION.md** - Visual diagrams and comparisons

### Notebooks Guide

See `notebooks/README.md` for:
- Workflow organization
- Notebook descriptions
- Setup instructions
- Tips and common patterns

## 📁 Project Structure

```
diffusion-offrl/
├── src/difsched/              ← Main package
│   ├── agents/                ← RL agents (DQL, DRL, MDP)
│   │   ├── dql/
│   │   ├── drl/
│   │   └── mdp/
│   ├── env/                   ← Environment simulation
│   ├── training/              ← Training pipelines
│   ├── evaluation/            ← Evaluation tools
│   ├── data/                  ← Data management
│   ├── config/                ← Configuration system
│   └── utils/                 ← Utilities
│
├── notebooks/                 ← Jupyter notebooks
│   ├── 00_getting_started/
│   ├── 01_data_pipeline/
│   ├── 02_training/
│   ├── 03_evaluation/
│   ├── 04_analysis/
│   └── 05_benchmarks/
│
├── configs/                   ← Configuration files
│   ├── env/
│   ├── models/
│   └── training/
│
├── data/                      ← Data storage
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── scripts/                   ← CLI utilities
├── docs/                      ← Documentation
├── setup.py                   ← Package installer
├── requirements.txt           ← Dependencies
└── .gitignore                 ← Version control
```

## 🔧 Configuration

### Environment Config

Edit `configs/env/default.yaml`:

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

### Load Configuration in Python

```python
from difsched.config import ConfigManager

cfg_mgr = ConfigManager(config_dir='configs')
env_config = cfg_mgr.load_env_config('default')
model_config = cfg_mgr.load_model_config('dql_default')
train_config = cfg_mgr.load_training_config('offline')
```

## 📖 Usage Examples

### Example 1: Load Configuration

```python
from difsched.config import EnvironmentConfig

# From YAML
config = EnvironmentConfig.from_yaml('configs/env/default.yaml')

# From dict
config = EnvironmentConfig.from_dict({
    'N_user': 8,
    'LEN_window': 200,
    'N_aggregation': 4,
    'dataflow': 'thumb_fr',
    'B': 100,
    'r_bar': 5,
})

# To dict
config_dict = config.to_dict()

# To YAML
config.to_yaml('configs/env/custom.yaml')
```

### Example 2: Use in Notebooks

```python
import sys
sys.path.insert(0, '../..')

from difsched import env, agents, config
from difsched.config import ConfigManager

# Setup
cfg_mgr = ConfigManager(config_dir='../../configs')
env_config = cfg_mgr.load_env_config('default')

# Create environment (when available)
# environment = env.Environment(env_config.to_dict(), traffic_gen)
```

## 🎯 Workflow

### Complete Pipeline

```
1. Setup Environment
   └─ Configure via configs/env/
   
2. Generate Data
   └─ notebooks/01_data_pipeline/
   
3. Train Models
   └─ notebooks/02_training/
   
4. Evaluate Performance
   └─ notebooks/03_evaluation/
   
5. Analyze Results
   └─ notebooks/04_analysis/
   
6. Run Benchmarks
   └─ notebooks/05_benchmarks/
```

## 🔄 Migration from Old Structure

If you're migrating from the old project structure:

1. **Old code** remains in original directories (Agents/, Environment/, etc.)
2. **New structure** is in `src/difsched/`
3. **Gradual migration**: Move modules one by one
4. **Update imports** as you migrate

See `IMPLEMENTATION_GUIDE.md` for detailed migration steps.

## 🛠️ Development

### Install Development Dependencies

```bash
pip install -e ".[dev,notebooks]"
```

### Code Structure Guidelines

- **agents/**: Agent implementations
- **env/**: Environment and simulation code
- **training/**: Training pipelines
- **evaluation/**: Metrics and visualization
- **data/**: Data management and processing
- **config/**: Configuration management
- **utils/**: General utilities

### Adding New Modules

1. Create directory in `src/difsched/`
2. Add `__init__.py`
3. Implement functionality
4. Update parent `__init__.py`
5. Add to documentation

## 📊 Data Management

### Data Directory Structure

```
data/
├── raw/                  ← Original data
│   ├── traffic/
│   └── generated/
├── processed/            ← Processed data
│   ├── offline_dataset/
│   ├── policies/
│   └── trained_models/
└── results/             ← Experiment results
    ├── evaluations/
    └── figures/
```

### Note

- `data/` directory is gitignored
- Use `data/raw/` for input data
- Generated files go to `data/processed/`
- Results saved to `data/results/`

## 🔍 Troubleshooting

### Import Errors

```python
# ✓ Correct
from difsched import env, agents
from difsched.config import ConfigManager

# ✗ Wrong
from Agents.DQL import dql  # Old structure
```

### Module Not Found

```bash
# Reinstall package
pip install -e .

# Verify installation
python -c "from difsched import agents; print(agents)"
```

### Configuration Not Found

```python
# Check config directory
from pathlib import Path
config_dir = Path('configs')
print(list(config_dir.glob('**/*.yaml')))
```

## 📝 Notebook Workflow

### Standard Notebook Setup

```python
# Cell 1: Setup path and imports
import sys
sys.path.insert(0, '../..')

from difsched import env, agents, config
from difsched.config import ConfigManager

# Cell 2: Load configuration
cfg_mgr = ConfigManager(config_dir='../../configs')
env_config = cfg_mgr.load_env_config('default')

# Cell 3: Create components
# (when available)

# Cell 4: Run experiments

# Cell 5: Analyze results
```

## 🧪 Methodology

### Diffusion Q-Learning (DQL)

The core innovation of this project is combining diffusion models with Q-learning for offline reinforcement learning:

- **Diffusion Policy**: Models action distributions using reverse diffusion processes
- **Q-Learning**: Learns optimal Q-functions for resource allocation decisions
- **Offline Learning**: Trains from fixed datasets without environment interaction
- **IQL Integration**: Uses Implicit Q-Learning for value function estimation

### Key Algorithms

1. **DQL with IQL**: Diffusion Q-Learning with Implicit Q-Learning critic
2. **Behavioral Cloning**: Initial policy learning from expert demonstrations
3. **Diffusion Schedule**: Stochastic noise schedule for action generation
4. **EMA Targets**: Exponential moving average targets for stable training

### Environment Dynamics

- **Wireless Network Simulation**: Realistic channel fading and interference
- **Traffic Patterns**: Time-varying user demands with prediction
- **Resource Constraints**: Limited bandwidth with QoS requirements
- **State Space**: User demands, predicted traffic, channel conditions

## 🚦 Status

### Implemented ✅
- [x] Diffusion Q-Learning algorithm (DQL)
- [x] Environment simulation with traffic prediction
- [x] Training pipelines for offline RL
- [x] Evaluation framework
- [x] Deep RL (SAC) and MDP baselines
- [x] Configuration system
- [x] Data pipeline for traffic generation
- [x] Notebook workflow for experiments

### Research Areas 🔬
- Diffusion-based policy learning for continuous action spaces
- Offline reinforcement learning from fixed datasets
- Resource allocation in wireless networks
- Multi-objective optimization (latency, throughput, reliability)

### Recent Improvements 🔄
- Organized package structure (`src/difsched/`)
- Centralized configuration (YAML-based)
- Improved evaluation metrics
- Better documentation

## 📚 References

### Academic Papers

**Diffusion Models in RL:**
- Diffusion Q-Learning for Offline Reinforcement Learning
- Implicit Q-Learning (IQL) by Kostrikov et al.

**Wireless Resource Allocation:**
- 5G Network Slicing and Resource Allocation
- Deep Reinforcement Learning for Wireless Network Management

**Offline RL:**
- Conservative Q-Learning for Offline RL
- One-Step Actor-Critic Methods

### Python Packaging
- [Python Packaging Guide](https://packaging.python.org/)
- [setuptools Documentation](https://setuptools.pypa.io/)

### Project Structure
- [Hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
- [Real Python - Project Structure](https://realpython.com/python-application-layouts/)

## 📄 License

MIT License - See LICENSE file for details

Copyright © 2024-2025 L2S Research Group. All rights reserved.

This software is provided for educational and research purposes. See LICENSE file for full terms and conditions.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{diffusion_offrl_2024,
  title = {Diffusion-based Resource Scheduling with Offline RL},
  author = {L2S Research Group},
  year = {2024-2025},
  url = {https://github.com/yourusername/diffusion-offrl}
}
```

## 👤 Authors & Contributors

L2S Research Group

**Primary Contributors:**
- Resource scheduling algorithms
- Diffusion-based RL implementation
- Evaluation framework

For questions or collaborations, please contact: research@l2s.group

## 🤝 Contributing

We welcome contributions! This is a research project and contributions can include:

### Types of Contributions
- **Algorithm improvements**: Better diffusion schedules, training techniques
- **New environments**: Additional wireless network scenarios
- **Baselines**: New RL algorithms for comparison
- **Evaluation metrics**: Additional performance measures
- **Documentation**: Code comments, tutorials, examples
- **Bug fixes**: Testing and debugging

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Follow the project structure and coding style
4. Update documentation as needed
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Development Guidelines
- Follow the existing code structure in `src/difsched/`
- Add docstrings to new functions and classes
- Update notebooks if applicable
- Test your changes before submitting
- Reference relevant papers in code comments

## 📞 Support

### Getting Help

1. Check relevant documentation files
2. See examples in notebooks/
3. Review project structure guide
4. Check implementation guide for migration issues

### Documentation Files

- **Quick questions** → README.md (this file)
- **Architecture questions** → EXECUTIVE_SUMMARY.md
- **Structure questions** → PROJECT_STRUCTURE_ANALYSIS.md
- **Migration help** → IMPLEMENTATION_GUIDE.md
- **Visual guide** → STRUCTURE_VISUALIZATION.md

---

**Status**: ✅ **READY FOR DEVELOPMENT**

Start with `notebooks/00_getting_started/` and follow the workflow!
