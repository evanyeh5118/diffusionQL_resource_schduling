# Diffusion-based Resource Scheduling with Offline RL

A minimal codebase for scheduling using offline reinforcement learning with diffusion-based policies.

## Project Structure

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
