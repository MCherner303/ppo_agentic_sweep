#!/bin/bash
# Script to rerun the best PPO config with 10 new seeds
top_config_args="--actor-lr 7e-05 --critic-lr 0.0001 --clip-param 0.15 --gae-lambda 0.95 --entropy-coef 0.005 --num-mini-batches 4 --ppo-epochs 4 --batch-size 2048 --num-steps 2048 --max-grad-norm 0.5 --value-loss-coef 0.5 --gamma 0.99 --eps 0.2"

mkdir -p logs/top_config_repeats
for SEED in 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010; do
  echo "Running top config with seed $SEED..."
  python train_ppo.py $top_config_args --seed $SEED > logs/top_config_repeats/top_config_seed_${SEED}.log 2>&1
done
