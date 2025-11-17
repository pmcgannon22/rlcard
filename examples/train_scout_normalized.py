''' Train Scout with NORMALIZED rewards (optional)

This script uses normalized rewards in [-1, 1] range instead of raw scores.
Use this if you want to:
- Apply standard RL hyperparameters from papers
- Improve training stability
- Compare different algorithms fairly

For most users, train_scout_improved_m1.py (unnormalized) is recommended.
'''
import os
import argparse
import torch
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcard.envs.scout_normalized import make_normalized_scout
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import (
    set_seed,
    evaluate_win_rate,
    Logger,
    plot_curve,
    reorganize_with_shaped_rewards,
)


def get_m1_device():
    """Get the best device for M1 Mac"""
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) available - using GPU acceleration")
        return torch.device("mps")
    else:
        print("⚠ MPS not available - falling back to CPU")
        return torch.device("cpu")


def train(args):
    # Get M1-optimized device
    device = get_m1_device()

    # Seed for reproducibility
    set_seed(args.seed)

    # Make NORMALIZED Scout environment
    env = make_normalized_scout({
        'reward_shaping': True,
        'reward_scale': 15.0,  # Max typical score
        'normalize_shaped_rewards': True,
    })

    print(f"\n{'='*60}")
    print("SCOUT TRAINING - NORMALIZED REWARDS")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"State shape: {env.state_shape[0]} (dense encoding)")
    print(f"Number of actions: {env.num_actions}")
    print(f"Reward range: [-1.0, +1.0] (normalized)")
    print(f"Improvements enabled:")
    print(f"  ✓ Dense encoding")
    print(f"  ✓ Reward normalization (NEW)")
    print(f"  ✓ Reward shaping (scaled)")
    print(f"  ✓ Action features")
    print(f"{'='*60}\n")

    # Initialize DQN agent
    if args.load_checkpoint_path != "":
        agent = DQNAgent.from_checkpoint(
            checkpoint=torch.load(args.load_checkpoint_path, map_location=device, weights_only=False)
        )
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
            batch_size=args.batch_size,
            replay_memory_size=args.replay_memory_size,
        )

    # Set up agents
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Training loop
    print(f"Starting training for {args.num_episodes} episodes...\n")

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            # Generate data
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize with shaped rewards
            trajectories = reorganize_with_shaped_rewards(
                trajectories,
                payoffs,
                env.shaped_rewards
            )

            # Train agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate
            if episode % args.evaluate_every == 0:
                eval_results = evaluate_win_rate(env, args.num_eval_games, player_id=0)
                logger.log_performance(episode, eval_results['avg_payoff'])

                print(f"Episode {episode:5d} | "
                      f"Win Rate: {eval_results['win_rate']:5.1f}% | "
                      f"Avg Reward: {eval_results['avg_payoff']:6.3f} | "
                      f"Epsilon: {agent.epsilons[0]:.3f}")

            # Detailed evaluation
            if episode > 0 and episode % (args.evaluate_every * 5) == 0:
                print(f"\n--- Detailed Evaluation at Episode {episode} ---")
                detailed = evaluate_win_rate(env, args.num_eval_games * 2, player_id=0)
                print(f"Win rate: {detailed['win_rate']:.1f}% "
                      f"({detailed['wins']}W-{detailed['losses']}L-{detailed['ties']}T)")
                print(f"Normalized reward: {detailed['avg_payoff']:.4f} (range: [-1, 1])")
                print("-" * 50 + "\n")

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot and save
    plot_curve(csv_path, fig_path, 'dqn')
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved: {save_path}")
    print(f"Note: This model expects NORMALIZED rewards")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Scout training with normalized rewards")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_eval_games', type=int, default=500)
    parser.add_argument('--evaluate_every', type=int, default=200)
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[256, 256, 128])
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epsilon_decay_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=15000)
    parser.add_argument('--log_dir', type=str, default='experiments/scout_normalized/')
    parser.add_argument('--load_checkpoint_path', type=str, default="")
    parser.add_argument('--save_every', type=int, default=2000)

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
