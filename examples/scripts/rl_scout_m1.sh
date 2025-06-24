#!/bin/bash

# DQN training for Scout on M1
python3 examples/run_rl.py \
    --env scout \
    --algorithm dqn \
    --num_episodes 15000 \
    --num_eval_games 2000 \
    --evaluate_every 300 \
    --log_dir experiments/scout_dqn_m1/ \
    --seed 42

echo "Training completed! Check experiments/scout_dqn_m1/ for results."