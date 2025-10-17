# Diffusion-based Resource Scheduling with Offline RL

A comprehensive framework for resource scheduling in wireless networks using diffusion-based offline reinforcement learning techniques.

## 🎯 Overview

This project implements advanced RL techniques for intra-slice resource scheduling in next-generation wireless networks. It includes:

- **Agents**: Diffusion Q-Learning (DQL), Deep RL, and MDP-based agents
- **Environment**: Realistic wireless network simulation with traffic generation
- **Training**: Offline reinforcement learning pipeline with behavioral cloning
- **Evaluation**: Comprehensive evaluation and analysis tools
- **Data Pipeline**: Automatic traffic dataset generation and processing

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

## 🚦 Status

### Implemented ✅
- [x] Directory structure
- [x] Package setup (setup.py)
- [x] Configuration system
- [x] Notebook organization
- [x] Documentation

### In Progress 🔄
- [ ] Module migrations from old structure
- [ ] Unit tests (skipped as requested)
- [ ] CI/CD pipeline

### TODO 📋
- [ ] DQL agent implementation
- [ ] DRL agent implementation
- [ ] MDP solver implementation
- [ ] Training pipelines
- [ ] Evaluation metrics
- [ ] Visualization tools

## 📚 References

### Python Packaging
- [Python Packaging Guide](https://packaging.python.org/)
- [setuptools Documentation](https://setuptools.pypa.io/)

### Project Structure
- [Hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
- [Real Python - Project Structure](https://realpython.com/python-application-layouts/)

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Your Name <your.email@example.com>

## 🤝 Contributing

Contributions welcome! Please:
1. Create feature branch
2. Follow project structure
3. Update documentation
4. Submit pull request

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
