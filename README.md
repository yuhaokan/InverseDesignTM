# InverseDesignTM

Inverse design of transmission matrices in 2D metallic billiard cavities using reinforcement learning (PPO/SAC) and MEEP electromagnetic simulations.

## Overview

This project uses deep reinforcement learning to optimize the positions of dielectric scatterers inside metallic billiard cavities, targeting specific transmission matrix (TM) properties such as:

- **Rank-1 TM** — singular-value dominated transmission
- **Rank-1 Trace-0** — rank-1 with vanishing trace
- **Degenerate Eigenvalues** — coalescing eigenvalues (exceptional point design)
- **Degenerate Singular Values** — equal singular values
- **Fixed Target** — match a specific complex TM

The RL agent iteratively adjusts scatterer positions to minimize an error metric derived from the TM, computed via full-wave FDTD simulations in [MEEP](https://meep.readthedocs.io/).


## Setup

### 1. Create conda environment

From lock file (recommended, exact reproducibility):

```bash
pip install conda-lock
conda-lock install conda-lock.yml -n meep
conda activate meep
```

Or from environment.yml (resolves latest compatible versions):

```bash
conda env create -f environment.yml
conda activate meep
```

### 2. Install the package in editable mode

From the project root (`~/workplace/InverseDesignTM`):

```bash
pip install -e .
```

This registers `inverse_design` as an importable package. Edits to source files take effect immediately — no reinstall needed.

### Regenerating the lock file (maintainers)

```bash
conda-lock -f environment.yml -p linux-64
```

Key dependencies: `meep` (FDTD), `stable-baselines3` (RL), `gymnasium`, `pytorch`, `numpy`, `matplotlib`, `tensorboard`.

## Usage

### Training

```bash
# Visualize policy only, no training
python -m inverse_design.solution --visualize

# Defaults: PPO, BilliardTwo, DegenerateEigVal
python -m inverse_design.solution

# Custom configuration
python -m inverse_design.solution --algo PPO --billiard BilliardTwo --target Rank1 --error-threshold 0.001 --n-envs 4
```

Options:
- `--algo`: `PPO` or `SAC` (default: `PPO`)
- `--billiard`: `BilliardTwo` or `BilliardThree` (default: `BilliardTwo`)
- `--target`: `Rank1`, `Rank1Trace0`, `DegenerateEigVal`, `FixedTarget`, `DegenerateSingularVal` (default: `Rank1`)
- `--error-threshold`: early stopping threshold (default: `0.02`)
- `--n-envs`: number of parallel environments (default: `4`)
- `--visualize`: visualize the network architecture and exit (no training)

### Single environment test

```bash
python -m inverse_design.envs.billiard_two_env
python -m inverse_design.envs.billiard_three_env
```


Best scatterer positions are saved to `positions/` during training. 

Monitor progress with:

```bash
tensorboard --logdir src/inverse_design/logs
```

See [TensorBoard PPO Metrics](src/inverse_design/docs/tensorboard_ppo_metrics.md) for a detailed explanation of each metric.

To see the final design result of training, see [validate.ipynb](src/inverse_design/validate.ipynb).

## How It Works

1. **Environment**: A metallic billiard cavity with waveguide ports is modeled in MEEP. Dielectric cylinders (scatterers) are placed inside. The agent's observation and action spaces are the normalized 2D positions of all scatterers.

2. **TM Calculation**: Meep calculates the scattering matrix. 

3. **Reward**: The error function (selected via `TargetType`) quantifies how far the current TM is from the desired property. The reward is the negative error.

4. **Training**: The RL agent (PPO or SAC via Stable-Baselines3) learns to adjust scatterer positions to minimize the error. Callbacks track and save the best configuration found.

## Citation

If you find this code helpful, please cite:

```bibtex
@article{kang2025inverse,
  title={Inverse design of the transmission matrix in a random system using Reinforcement Learning},
  author={Yuhao Kang},
  year={2025},
  eprint={2506.13057},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2506.13057}
}
```
