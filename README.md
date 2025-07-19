# PPO Agentic Hyperparameter Sweep

> **Robustness Evaluation:**
> See [`ROBUSTNESS_EVAL.md`](ROBUSTNESS_EVAL.md) for a deep-dive on multi-seed robustness evaluation, technical stability tricks, and the latest results on top configs.

A robust framework for running large-scale hyperparameter sweeps with Proximal Policy Optimization (PPO) agents in custom gridworld environments.

## Overview
This repository showcases a reproducible, agentic approach to hyperparameter optimization for PPO reinforcement learning. It features:

- **Configurable PPO agent and environment wrappers** (multi-agent gridworld, PettingZoo-compatible)
- **Automated hyperparameter grid search** with multi-seed averaging, early stopping, and periodic greedy evaluation
- **Robust subprocess management**: detects and logs failed runs, captures errors, and prevents silent failures
- **Structured logging and result saving** for easy analysis and reproducibility

## Key Features
- **Hyperparameter Sweep Script**: `train/improved_ppo_sweep.py`
    - Defines and runs a grid search over PPO hyperparameters (learning rates, clip, GAE λ, entropy, etc.)
    - Runs each config with multiple random seeds
    - Early-stops unproductive runs
    - Logs all results to `sweep_logs/` for later analysis
- **PPO Training Script**: `train/train_ppo.py`
    - Modular, curriculum-compatible PPO trainer
    - Logs per-episode rewards in a sweep-parsable format
    - Supports evaluation, checkpointing, and curriculum learning
- **Agent & Environment**: Modular design for custom environments and agents (see `agents/`, `environment/`)
- **Reproducibility**: All random seeds, configs, and results are saved

## Example Usage

```bash
# Run a sweep with 3 seeds per config
python train/improved_ppo_sweep.py --log-dir sweep_logs/my_sweep --num-seeds 3

# Run a single config for debugging
python train/improved_ppo_sweep.py --max-configs 1 --num-seeds 1
```

## Results
- Sweep logs and results are saved to `sweep_logs/` (excluded from git).
- Once your sweep completes, you can analyze `all_results.json` and upload summary plots to this repo.

## Repo Structure
- `train/improved_ppo_sweep.py` — Hyperparameter sweep orchestrator
- `train/train_ppo.py` — PPO training script
- `agents/` — PPO agent and utilities
- `environment/` — Environment wrappers
- `requirements.txt` — Python dependencies
- `.gitignore` — Excludes logs, outputs, and virtual envs

## About
Created by [MCherner303](https://github.com/MCherner303) to demonstrate advanced RL experimentation and sweep automation.

---

**Pro tip:** After your sweep finishes, add your best configs and analysis plots to this repo for maximum impact!
