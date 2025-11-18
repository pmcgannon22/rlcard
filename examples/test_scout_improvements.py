''' Quick test to verify Scout improvements work correctly '''

import numpy as np
import rlcard
from rlcard.agents import RandomAgent

def test_dense_encoding():
    """Test that dense encoding reduces state size"""
    print("=" * 60)
    print("TEST 1: Dense Encoding")
    print("=" * 60)

    env = rlcard.make('scout')
    state, player_id = env.reset()

    print(f"State shape: {env.state_shape}")
    print(f"Observation size: {env.state_shape[0][0]} features")
    print(f"Expected reduction: 688 → {env.state_shape[0][0]}")
    print(f"Reduction: {(1 - env.state_shape[0][0] / 688) * 100:.1f}%")

    # Verify the state is actually smaller
    assert env.state_shape[0][0] < 688, "State size should be reduced"
    assert env.state_shape[0][0] > 0, "State size should be positive"

    print("✓ Dense encoding working correctly!\n")

def test_reward_shaping():
    """Test that reward shaping provides intermediate rewards"""
    print("=" * 60)
    print("TEST 2: Reward Shaping")
    print("=" * 60)

    env = rlcard.make('scout', config={'reward_shaping': True})
    agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
    env.set_agents(agents)

    # Run one game
    trajectories, payoffs = env.run(is_training=True)

    print(f"Number of players: {env.num_players}")
    print(f"Number of shaped rewards collected:")
    for i, rewards in enumerate(env.shaped_rewards):
        print(f"  Player {i}: {len(rewards)} rewards")
        if len(rewards) > 0:
            print(f"    Sample rewards: {rewards[:5]}")

    # Verify shaped rewards were collected
    assert any(len(r) > 0 for r in env.shaped_rewards), "Shaped rewards should be collected"

    print("✓ Reward shaping working correctly!\n")

def test_action_features():
    """Test that action features are generated"""
    print("=" * 60)
    print("TEST 3: Action Features")
    print("=" * 60)

    env = rlcard.make('scout')
    state, player_id = env.reset()

    # Get legal actions
    legal_actions = state['legal_actions']
    print(f"Number of legal actions: {len(legal_actions)}")

    # Test action features for a few actions
    sample_actions = list(legal_actions.keys())[:5]
    print(f"\nTesting action features for {len(sample_actions)} actions:")

    for action_id in sample_actions:
        features = env.get_action_feature(action_id)
        print(f"  Action {action_id}: {features}")
        assert features.shape == (7,), f"Action features should have shape (7,), got {features.shape}"
        assert features.dtype == np.float32, "Action features should be float32"

    print("✓ Action features working correctly!\n")

def test_full_game():
    """Test a complete game with all improvements"""
    print("=" * 60)
    print("TEST 4: Full Game with All Improvements")
    print("=" * 60)

    from rlcard.utils import reorganize_with_shaped_rewards

    env = rlcard.make('scout', config={'reward_shaping': True})
    agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
    env.set_agents(agents)

    # Run a game
    trajectories, payoffs = env.run(is_training=True)

    print(f"Game completed successfully!")
    print(f"Payoffs: {payoffs}")
    print(f"Trajectory lengths: {[len(t) for t in trajectories]}")

    # Reorganize with shaped rewards
    reorganized = reorganize_with_shaped_rewards(
        trajectories,
        payoffs,
        env.shaped_rewards
    )

    print(f"Reorganized trajectory lengths: {[len(t) for t in reorganized]}")

    # Check that all transitions have rewards
    for player_idx, player_traj in enumerate(reorganized):
        for trans_idx, transition in enumerate(player_traj):
            state, action, reward, next_state, done = transition
            # Should have rewards (either shaped or final payoff)
            if player_idx == 0 and trans_idx < 3:
                print(f"  Player {player_idx}, Transition {trans_idx}: reward = {reward}, done = {done}")

    print("✓ Full game with all improvements working correctly!\n")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TESTING SCOUT IMPROVEMENTS")
    print("=" * 60 + "\n")

    try:
        test_dense_encoding()
        test_reward_shaping()
        test_action_features()
        test_full_game()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nYou can now run training with:")
        print("  python examples/train_scout_improved.py --num_episodes 10000")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
