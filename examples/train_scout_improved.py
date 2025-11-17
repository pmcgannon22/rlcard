''' Train Scout with improved state representation and reward shaping

This script demonstrates the high-ROI improvements:
1. Dense encoding (instead of one-hot) - reduces state size by ~83%
2. Reward shaping - provides intermediate feedback
3. Action features - helps network understand action semantics
'''
import os
import argparse
import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize_with_shaped_rewards,
    Logger,
    plot_curve,
)

def train(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with reward shaping enabled
    env = rlcard.make(
        'scout',
        config={
            'seed': args.seed,
            'reward_shaping': True,  # Enable reward shaping
        }
    )

    print(f"Environment created:")
    print(f"  State shape: {env.state_shape}")
    print(f"  Number of actions: {env.num_actions}")
    print(f"  Reward shaping: ENABLED")
    print(f"  Dense encoding: ENABLED (state size reduced from 688 to {env.state_shape[0][0]})")

    # Initialize the DQN agent
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        if args.load_checkpoint_path != "":
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path, weights_only=False))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=args.mlp_layers,
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every,
                learning_rate=args.learning_rate,
                epsilon_decay_steps=args.epsilon_decay_steps,
            )

    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        if args.load_checkpoint_path != "":
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path, weights_only=False))
        else:
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[128, 128],
                q_mlp_layers=[128, 128],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    # Set up agents (learning agent + random opponents)
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    print(f"\nStarting training for {args.num_episodes} episodes...")
    print(f"  Evaluating every {args.evaluate_every} episodes")
    print(f"  Using {args.algorithm.upper()} agent")
    print(f"  Device: {device}")

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Use shaped rewards for reorganization
            if env.use_reward_shaping:
                trajectories = reorganize_with_shaped_rewards(
                    trajectories,
                    payoffs,
                    env.shaped_rewards
                )
            else:
                from rlcard.utils import reorganize
                trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory and train
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance
            if episode % args.evaluate_every == 0:
                avg_reward = tournament(env, args.num_eval_games)[0]
                logger.log_performance(episode, avg_reward)
                print(f"Episode {episode}: Average reward = {avg_reward:.4f}")

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print(f'\nModel saved in {save_path}')
    print(f'Training complete!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Scout training with improvements")
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=['dqn', 'nfsp'],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10000,
        help='Number of training episodes (default: 10000)'
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=1000,
        help='Number of games for evaluation (default: 1000)'
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=500,
        help='Evaluate every N episodes (default: 500)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/scout_improved/',
        help='Directory to save logs and models'
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
        help='Path to load checkpoint from'
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=2000,
        help='Save checkpoint every N episodes'
    )
    parser.add_argument(
        '--mlp_layers',
        type=int,
        nargs='+',
        default=[256, 256, 128],
        help='MLP layer sizes (default: 256 256 128)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--epsilon_decay_steps',
        type=int,
        default=30000,
        help='Epsilon decay steps (default: 30000)'
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
