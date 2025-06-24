''' An example of training a Deep Monte-Carlo (DMC) Agent on Scout for M1 Macs
'''
import os
import argparse
import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

def train(args):
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) for training")
        device = "mps"
    else:
        print("MPS not available, falling back to CPU")
        device = "cpu"

    # Make the environment
    env = rlcard.make('scout')

    # Initialize the DMC trainer
    trainer = DMCTrainer(
        env,
        cuda='',  # No CUDA on M1
        load_model=args.load_model,
        xpid=args.xpid,
        savedir=args.savedir,
        save_interval=args.save_interval,
        num_actor_devices=1,  # M1 has limited GPU cores
        num_actors=args.num_actors,
        training_device=device,
    )

    # Train DMC Agents
    trainer.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DMC Scout training for M1 Macs")
    parser.add_argument(
        '--load_model',
        action='store_true',
        help='Load an existing model',
    )
    parser.add_argument(
        '--xpid',
        default='scout_m1',
        help='Experiment id (default: scout_m1)',
    )
    parser.add_argument(
        '--savedir',
        default='experiments/dmc_scout_m1',
        help='Root dir where experiment data will be saved'
    )
    parser.add_argument(
        '--save_interval',
        default=30,
        type=int,
        help='Time interval (in minutes) at which to save the model',
    )
    parser.add_argument(
        '--num_actors',
        default=4,  # Reduced for M1
        type=int,
        help='The number of actors for simulation (reduced for M1)',
    )

    args = parser.parse_args()

    # Don't set CUDA_VISIBLE_DEVICES on M1
    train(args)