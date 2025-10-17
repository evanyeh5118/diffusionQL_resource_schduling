# Implementation Guide: Restructuring Your Project

This guide provides step-by-step instructions to transform your project from scattered structure to organized, production-ready codebase.

---

## Quick Summary

| Current | Target | Time |
|---------|--------|------|
| 🔴 Scattered modules | 🟢 `src/difsched/` package | ~2-3 weeks |
| 🔴 Mixed notebooks | 🟢 Organized by workflow | |
| 🔴 Manual path setup | 🟢 `pip install -e .` | |
| 🔴 No tests | 🟢 Test structure | |
| 🔴 Scattered data | 🟢 `data/` directory | |

---

## Phase 1: Preparation & Planning (Day 1)

### Step 1: Backup & Version Control
```bash
# Create backup branch
git checkout -b refactor/project-restructure
git push -u origin refactor/project-restructure

# Or backup manually
cp -r . ../diffusion-offrl-backup-$(date +%Y%m%d)
```

### Step 2: Create Project Structure
```bash
# Create new directory structure
mkdir -p src/difsched/{config,env,agents,training,evaluation,data,utils}
mkdir -p src/difsched/agents/{dql,drl,mdp}
mkdir -p tests/{unit,integration}
mkdir -p notebooks/{00_getting_started,01_data_pipeline,02_training,03_evaluation,04_analysis,05_benchmarks}
mkdir -p configs/{env,models,training}
mkdir -p scripts
mkdir -p docs
mkdir -p data/{raw,processed,results}
mkdir -p data/{raw/traffic,processed/offline_dataset,processed/policies,processed/trained_models,results/evaluations,results/figures}
```

### Step 3: Create Essential Files

#### `.gitignore`
```
# Data files
data/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
ENV/
env/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
```

#### `requirements.txt`
```
# Core dependencies (extract from your imports)
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
torch>=1.10.0
torchvision>=0.11.0
torchaudio>=0.10.0

# Jupyter & notebooks
jupyter>=1.0.0
ipython>=7.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Development
pytest>=6.2.0
pytest-cov>=2.12.0
black>=21.0
flake8>=3.9.0
mypy>=0.910

# Configuration
pyyaml>=5.4
pydantic>=1.8

# Utilities
tqdm>=4.62.0
```

#### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="difsched",
    version="0.1.0",
    description="Diffusion-based Resource Scheduling with Offline RL",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/diffusion-offrl",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",
        "pyyaml>=5.4",
        "pydantic>=1.8",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
)
```

#### `README.md` (Comprehensive)
```markdown
# Diffusion-based Resource Scheduling with Offline RL

## Overview

A comprehensive framework for resource scheduling in wireless networks using 
diffusion-based offline reinforcement learning techniques.

## Features

- **Agents**: DQL, DRL, and MDP-based agents
- **Environment**: Realistic wireless network simulation
- **Training**: Offline RL training pipeline
- **Evaluation**: Comprehensive evaluation and analysis tools
- **Data Pipeline**: Automatic traffic dataset generation and processing

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/diffusion-offrl.git
cd diffusion-offrl

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### First Steps

```bash
# Validate environment setup
python -c "from difsched import env, agents; print('Setup OK')"

# Run quick test
cd notebooks/00_getting_started
jupyter notebook 01_setup_environment.ipynb
```

## Project Structure

```
src/difsched/
├── agents/          # RL agents (DQL, DRL, MDP)
├── env/             # Environment simulation
├── training/        # Training pipelines
├── evaluation/      # Evaluation tools
├── data/            # Data management
├── config/          # Configuration system
└── utils/           # Utilities

notebooks/
├── 00_getting_started/
├── 01_data_pipeline/
├── 02_training/
├── 03_evaluation/
├── 04_analysis/
└── 05_benchmarks/
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Training Guide](docs/TRAINING_GUIDE.md)

## License

MIT License - See LICENSE file for details
```

---

## Phase 2: Migrate Core Modules (Days 2-5)

### Step 1: Create Package Init Files

#### `src/difsched/__init__.py`
```python
"""
Diffusion-based Resource Scheduling with Offline RL
"""

__version__ = "0.1.0"

from . import agents
from . import env
from . import training
from . import evaluation
from . import data
from . import config
from . import utils

__all__ = [
    "agents",
    "env", 
    "training",
    "evaluation",
    "data",
    "config",
    "utils",
]
```

#### `src/difsched/config/__init__.py`
```python
from .base_config import *
from .env_configs import *
from .model_configs import *

__all__ = [
    "EnvironmentConfig",
    "ModelConfig",
    "TrainingConfig",
    # Add more as needed
]
```

#### `src/difsched/agents/__init__.py`
```python
from .base import BaseAgent
from .dql import DQLAgent
from .drl import DRLAgent
from .mdp import MDPAgent

__all__ = [
    "BaseAgent",
    "DQLAgent",
    "DRLAgent",
    "MDPAgent",
]
```

### Step 2: Migrate Environment Module

**`src/difsched/env/__init__.py`**
```python
from .environment import Environment
from .traffic import TrafficGenerator
from .simulators import SimulatorType1
from .models import WirelessModel
from .rewards import RewardKernel

__all__ = [
    "Environment",
    "TrafficGenerator",
    "SimulatorType1",
    "WirelessModel",
    "RewardKernel",
]
```

**`src/difsched/env/environment.py`**
```python
# Copy from: Environment/EnvironmentSim.py
# Update imports to use relative paths
# Example:
import pickle
import numpy as np

from .simulators import SimulatorType1
from .traffic import TrafficGenerator

class Environment:
    def __init__(self, params, trafficGenerator):
        # ... existing code ...
        pass

def createEnv(envParams, trafficDataParentPath):
    # ... existing code ...
    pass
```

### Step 3: Checklist for Each Module

For each major module in `Agents/`, `DiffusionQL/`, and `Environment/`:

- [ ] Identify all source files
- [ ] Understand dependencies
- [ ] Create target directory in `src/difsched/`
- [ ] Copy files to new location
- [ ] Update all imports
- [ ] Update `__init__.py` files
- [ ] Test functionality
- [ ] Update path references in notebooks

### Step 4: Create Module Mapping Script

**`scripts/validate_imports.py`**
```python
#!/usr/bin/env python
"""Validate that all imports work after migration."""

import sys
import importlib

modules_to_test = [
    "difsched.env",
    "difsched.agents",
    "difsched.training",
    "difsched.evaluation",
    "difsched.data",
    "difsched.config",
]

failures = []
for module in modules_to_test:
    try:
        importlib.import_module(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")
        failures.append((module, e))

if failures:
    print(f"\n{len(failures)} module(s) failed to import")
    sys.exit(1)
else:
    print("\nAll modules imported successfully!")
    sys.exit(0)
```

---

## Phase 3: Configure & Centralize (Days 6-8)

### Step 1: Create Configuration System

**`src/difsched/config/base_config.py`**
```python
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import yaml
from pathlib import Path

@dataclass
class EnvironmentConfig:
    N_user: int
    LEN_window: int
    N_aggregation: int
    dataflow: str
    B: int
    r_bar: float
    randomSeed: int = 999
    sigmoid_k_list: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    sigmoid_s_list: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0])
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EnvironmentConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('environment', {}))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        """Load configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump({'environment': self.to_dict()}, f, default_flow_style=False)

@dataclass
class ModelConfig:
    """Base configuration for all agents."""
    name: str
    agent_type: str  # 'dql', 'drl', or 'mdp'
    learning_rate: float = 1e-4
    batch_size: int = 32
    hidden_dim: int = 256
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('model', {}))

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_episodes: int = 1000
    num_steps_per_episode: int = 200
    replay_buffer_size: int = 10000
    gamma: float = 0.99
    target_update_freq: int = 100
    log_freq: int = 10
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('training', {}))

class ConfigManager:
    """Central configuration manager."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
    
    def load_env_config(self, name: str = "default") -> EnvironmentConfig:
        path = self.config_dir / f"env/{name}.yaml"
        return EnvironmentConfig.from_yaml(str(path))
    
    def load_model_config(self, name: str = "default") -> ModelConfig:
        path = self.config_dir / f"models/{name}.yaml"
        return ModelConfig.from_yaml(str(path))
    
    def load_training_config(self, name: str = "default") -> TrainingConfig:
        path = self.config_dir / f"training/{name}.yaml"
        return TrainingConfig.from_yaml(str(path))
```

### Step 2: Create YAML Configuration Files

**`configs/env/default.yaml`**
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

**`configs/models/dql_default.yaml`**
```yaml
model:
  name: DQL_Default
  agent_type: dql
  learning_rate: 0.0001
  batch_size: 32
  hidden_dim: 256
```

**`configs/training/offline.yaml`**
```yaml
training:
  num_episodes: 1000
  num_steps_per_episode: 200
  replay_buffer_size: 10000
  gamma: 0.99
  target_update_freq: 100
  log_freq: 10
```

---

## Phase 4: Reorganize Notebooks (Days 9-11)

### Step 1: Create Notebook Index

**`notebooks/README.md`**
```markdown
# Notebooks Guide

## Workflow

1. **Getting Started** (`00_getting_started/`)
   - Setup and validate environment
   - Basic interface demonstration

2. **Data Pipeline** (`01_data_pipeline/`)
   - Generate traffic datasets
   - Train traffic predictor
   - Create offline dataset

3. **Training** (`02_training/`)
   - Train Diffusion Q-Learning agents
   - Train DRL agents
   - Train MDP policies

4. **Evaluation** (`03_evaluation/`)
   - Evaluate agent performance
   - Compare different agents
   - Generate benchmark results

5. **Analysis** (`04_analysis/`)
   - Analyze training dynamics
   - Visualize policy distributions
   - Model accuracy evaluation

6. **Benchmarks** (`05_benchmarks/`)
   - Run comprehensive experiments
   - Compare all methods

## Running Notebooks

```bash
cd notebooks
jupyter notebook 00_getting_started/01_setup_environment.ipynb
```
```

### Step 2: Create Notebook Templates

**`notebooks/00_getting_started/01_setup_environment.ipynb`**
```python
# Cell 1
import sys
sys.path.insert(0, '../..')

# Cell 2
from difsched import env, agents, config
from difsched.config import ConfigManager

print("✓ difsched package loaded successfully!")

# Cell 3
# Load configuration
cfg_mgr = ConfigManager(config_dir='../../configs')
env_config = cfg_mgr.load_env_config('default')
print(f"Environment config: {env_config}")

# Cell 4
# Test environment creation
environment = env.Environment(env_config.__dict__, None)
print("✓ Environment created successfully!")
```

---

## Phase 5: Add Tests (Days 12-13)

### Step 1: Create Test Structure

**`tests/conftest.py`**
```python
import pytest
from pathlib import Path

@pytest.fixture
def data_dir():
    return Path(__file__).parent.parent / "data"

@pytest.fixture
def config_dir():
    return Path(__file__).parent.parent / "configs"
```

**`tests/unit/test_config.py`**
```python
from difsched.config import EnvironmentConfig, ConfigManager

def test_env_config_creation():
    config = EnvironmentConfig(
        N_user=8,
        LEN_window=200,
        N_aggregation=4,
        dataflow="thumb_fr",
        B=100,
        r_bar=5
    )
    assert config.N_user == 8

def test_config_to_dict():
    config = EnvironmentConfig(
        N_user=8,
        LEN_window=200,
        N_aggregation=4,
        dataflow="thumb_fr",
        B=100,
        r_bar=5
    )
    d = config.to_dict()
    assert d['N_user'] == 8
    assert d['dataflow'] == 'thumb_fr'
```

### Step 2: Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/difsched --cov-report=html

# Run specific test file
pytest tests/unit/test_config.py -v
```

---

## Phase 6: Documentation (Days 14)

### Step 1: Create Key Docs

**`docs/GETTING_STARTED.md`**
```markdown
# Getting Started

## Installation

## Basic Usage

## Configuration

## Common Tasks
```

**`docs/ARCHITECTURE.md`**
```markdown
# Project Architecture

## Module Overview

## Data Flow

## Agent Design

## Training Pipeline
```

### Step 2: Update Module Docstrings

Add docstrings to all major classes and functions:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description
    
    Raises:
        ValueError: When this happens
    
    Example:
        >>> my_function(1, "test")
        True
    """
    pass
```

---

## Validation Checklist

After each phase, verify:

### Phase 1: Planning
- [ ] Backup created
- [ ] Git branch created
- [ ] All directories created
- [ ] Essential files created

### Phase 2: Module Migration
- [ ] All modules copied to `src/difsched/`
- [ ] All `__init__.py` files updated
- [ ] Imports validated with script
- [ ] All tests pass

### Phase 3: Configuration
- [ ] Configuration system works
- [ ] YAML files load correctly
- [ ] Config manager accessible

### Phase 4: Notebooks
- [ ] Notebooks reorganized
- [ ] All notebook imports updated
- [ ] README created

### Phase 5: Tests
- [ ] Test structure created
- [ ] Unit tests pass
- [ ] Coverage > 70%

### Phase 6: Documentation
- [ ] README comprehensive
- [ ] Architecture documented
- [ ] API documented

---

## Common Issues & Solutions

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'difsched'`

**Solution**:
```bash
pip install -e .
```

### Path Issues

**Problem**: Can't find data files

**Solution**: Use absolute paths from project root
```python
from pathlib import Path
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "raw"
```

### Circular Imports

**Problem**: `ImportError: cannot import name 'X' from partially initialized module`

**Solution**: Reorganize imports, avoid importing at module level:
```python
# Bad
from .agents import Agent  # In __init__.py

# Good
def get_agent():
    from .agents import Agent
    return Agent()
```

---

## Final Steps

### 1. Commit Changes
```bash
git add -A
git commit -m "refactor: restructure project to src/difsched package"
```

### 2. Test Everything
```bash
pytest tests/ -v
python scripts/validate_imports.py
```

### 3. Merge to Main
```bash
git checkout main
git pull origin main
git merge refactor/project-restructure
git push origin main
```

### 4. Update Documentation
- [ ] Update team documentation
- [ ] Create migration guide for collaborators
- [ ] Archive old structure as reference
- [ ] Update CI/CD pipelines

---

## Success Indicators

✅ Project successfully restructured when:

1. Code can be installed: `pip install -e .`
2. Imports work: `from difsched import agents, env, training`
3. Tests pass: `pytest tests/`
4. Notebooks run without path setup
5. Configuration system works
6. Documentation is comprehensive
7. No circular imports or missing modules
