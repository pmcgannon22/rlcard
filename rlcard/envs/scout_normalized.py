"""
Normalized reward wrapper for Scout environment

This wrapper normalizes Scout's raw point-based payoffs to [-1, 1] range
for more stable RL training. Use this if you're experiencing training
instability or want to use standard RL hyperparameters.

Usage:
    from rlcard.envs.scout_normalized import NormalizedScoutEnv
    env = NormalizedScoutEnv(config={'reward_shaping': True})
"""

import numpy as np
import rlcard
from rlcard.envs.scout import ScoutEnv


class NormalizedScoutEnv(ScoutEnv):
    """Scout environment with normalized rewards in [-1, 1] range.

    Normalizes based on typical Scout score ranges:
    - Max win: ~15 points → +1.0
    - Tie: 0 points → 0.0
    - Max loss: ~-15 points → -1.0

    Also scales shaped rewards proportionally to maintain consistency.
    """

    def __init__(self, config=None):
        config = config or {}
        super().__init__(config)

        # Typical Scout score range (adjust based on observed data)
        self.reward_scale = config.get('reward_scale', 15.0)  # Max typical score
        self.normalize_shaped_rewards = config.get('normalize_shaped_rewards', True)

        print(f"Reward normalization enabled (scale: ±{self.reward_scale})")

    def get_payoffs(self):
        """Get normalized payoffs in [-1, 1] range."""
        raw_payoffs = super().get_payoffs()

        # Normalize to [-1, 1]
        normalized = np.array(raw_payoffs) / self.reward_scale

        # Clip to prevent outliers
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def _compute_shaped_reward(self, prev_state, action, next_state):
        """Compute shaped reward with optional normalization."""
        raw_shaped_reward = super()._compute_shaped_reward(prev_state, action, next_state)

        if self.normalize_shaped_rewards:
            # Scale shaped rewards to match normalized payoff scale
            # Original shaped rewards are ~[-0.5, 3.0], normalize to ~[-0.1, 0.2]
            shaped_scale = 15.0  # Match reward_scale
            normalized_shaped = raw_shaped_reward / shaped_scale

            # Clip to reasonable range
            normalized_shaped = np.clip(normalized_shaped, -0.5, 0.5)

            return normalized_shaped
        else:
            return raw_shaped_reward


def make_normalized_scout(config=None):
    """Factory function to create normalized Scout environment.

    Args:
        config (dict): Configuration including:
            - reward_scale: Max score for normalization (default: 15.0)
            - normalize_shaped_rewards: Whether to normalize shaped rewards (default: True)
            - All standard Scout config options

    Returns:
        NormalizedScoutEnv: Scout environment with normalized rewards
    """
    return NormalizedScoutEnv(config or {})


if __name__ == '__main__':
    # Test the normalized environment
    print("Testing normalized Scout environment...")

    env = make_normalized_scout({'reward_shaping': True})

    from rlcard.agents import RandomAgent
    agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
    env.set_agents(agents)

    # Run a few games and check reward ranges
    print("\nRunning 100 test games...")
    payoffs_list = []

    for i in range(100):
        _, payoffs = env.run(is_training=False)
        payoffs_list.append(payoffs[0])

    payoffs_array = np.array(payoffs_list)

    print(f"\nNormalized Payoff Statistics:")
    print(f"  Min:  {payoffs_array.min():.4f}")
    print(f"  Max:  {payoffs_array.max():.4f}")
    print(f"  Mean: {payoffs_array.mean():.4f}")
    print(f"  Std:  {payoffs_array.std():.4f}")
    print(f"\n✓ All payoffs in [-1, 1]: {payoffs_array.min() >= -1 and payoffs_array.max() <= 1}")
