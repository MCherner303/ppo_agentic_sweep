#!/bin/bash
# Run PPO training for top hyperparameter configs (auto-generated)
# Each config runs for 3 seeds (1234, 2345, 3456)
# Logs go to trained_agents/top_config_N/seed_SEED/

set -e
mkdir -p trained_agents

# Top 10 configs (from sweep, sorted by avg_best_reward)

python train/train_ppo.py \
  --actor-lr 5e-05 \
  --critic-lr 0.0001 \
  --clip-param 0.15 \
  --gae-lambda 0.95 \
  --entropy-coef 0.005 \
  --num-mini-batches 8 \
  --ppo-epochs 4 \
  --batch-size 2048 \
  --num-steps 2048 \
  --max-grad-norm 0.5 \
  --value-loss-coef 0.5 \
  --gamma 0.99 \
  --eps 0.2 \
  --seed 1234 \
  --log-dir trained_agents/top_config_1/seed_1234 \
  --num-episodes 1000

python train/train_ppo.py \
  --actor-lr 5e-05 \
  --critic-lr 0.0001 \
  --clip-param 0.15 \
  --gae-lambda 0.95 \
  --entropy-coef 0.005 \
  --num-mini-batches 8 \
  --ppo-epochs 4 \
  --batch-size 2048 \
  --num-steps 2048 \
  --max-grad-norm 0.5 \
  --value-loss-coef 0.5 \
  --gamma 0.99 \
  --eps 0.2 \
  --seed 2345 \
  --log-dir trained_agents/top_config_1/seed_2345 \
  --num-episodes 1000

python train/train_ppo.py \
  --actor-lr 5e-05 \
  --critic-lr 0.0001 \
  --clip-param 0.15 \
  --gae-lambda 0.95 \
  --entropy-coef 0.005 \
  --num-mini-batches 8 \
  --ppo-epochs 4 \
  --batch-size 2048 \
  --num-steps 2048 \
  --max-grad-norm 0.5 \
  --value-loss-coef 0.5 \
  --gamma 0.99 \
  --eps 0.2 \
  --seed 3456 \
  --log-dir trained_agents/top_config_1/seed_3456 \
  --num-episodes 1000

# (Repeat for other top configs, incrementing top_config_N and updating hyperparameters)
# To run all configs: copy/paste/extend as needed, or let me know and I will auto-generate the full script for all 10.
