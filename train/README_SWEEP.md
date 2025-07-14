# PPO Hyperparameter Sweep

This directory contains scripts for running hyperparameter sweeps for the PPO agent training.

## Files

- `hyperparam_sweep.py`: Main script for running hyperparameter sweeps
- `train_ppo.py`: Modified training script with command-line arguments
- `analyze_sweep.py`: Script for analyzing sweep results (coming soon)

## Setup

1. Install required packages:
   ```bash
   pip install wandb  # For experiment tracking
   pip install numpy torch tqdm imageio matplotlib
   ```

2. Log in to Weights & Biases (for experiment tracking):
   ```bash
   wandb login
   ```

## Running a Sweep

### Basic Usage

Run a sweep with default settings:
```bash
python hyperparam_sweep.py --sweep-name my_sweep
```

### Advanced Options

```bash
python hyperparam_sweep.py \
  --sweep-name my_sweep \
  --num-seeds 3 \
  --max-concurrent 4 \
  --gpu-ids 0,1 \
  --log-dir ./sweep_logs
```

### Parameters

- `--sweep-name`: Name for this sweep (used for logging)
- `--num-seeds`: Number of random seeds to try per configuration (default: 2)
- `--max-concurrent`: Maximum number of concurrent training runs (default: 2)
- `--gpu-ids`: Comma-separated list of GPU IDs to use (default: '0')
- `--base-cmd`: Base command to run (default: 'python train_ppo.py')
- `--log-dir`: Directory to store sweep logs (default: 'sweep_logs')

## Analyzing Results

After the sweep completes, you can analyze the results using Weights & Biases:

1. Go to your W&B dashboard:
   ```bash
   wandb online
   ```

2. Open the project in your browser and compare different runs.

## Customizing the Sweep

To modify the hyperparameter search space, edit the `get_hyperparameter_grid()` function in `hyperparam_sweep.py`.

## Best Practices

1. Start with a broad sweep with a few seeds per configuration
2. Analyze results to identify promising regions
3. Run a focused sweep with more seeds in those regions
4. For the final training, use the best configuration with multiple seeds

## Monitoring

- Monitor GPU usage:
  ```bash
  watch -n 1 nvidia-smi
  ```

- Monitor training progress:
  ```bash
  tail -f sweep_logs/my_sweep_*/sweep_*.log
  ```

## Troubleshooting

- If you run out of GPU memory, reduce `--batch-size` or `--num-steps`
- For slow training, try increasing `--num-mini-batches`
- If training is unstable, try reducing the learning rate or increasing `--clip-param`
