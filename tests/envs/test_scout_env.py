import unittest
import numpy as np

import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.games.scout.utils import get_action_list, ScoutEvent
from .determism_util import is_deterministic

ACTION_LIST = get_action_list()
ACTION_FROM_ID = {v: k for k, v in ACTION_LIST.items()}

class TestUnoEnv(unittest.TestCase):

    def test_reset_and_extract_state(self):
        env = rlcard.make('scout')
        state, _ = env.reset()
        self.assertEqual(state['obs'].size, 2 * 12 * 12)

    def test_is_deterministic(self):
        self.assertTrue(is_deterministic('scout'))

    def test_get_legal_actions(self):
        env = rlcard.make('scout')
        env.set_agents([RandomAgent(env.num_actions) for _ in range(env.num_players)])
        env.reset()
        legal_actions = env._get_legal_actions()
        for legal_action in legal_actions:
            self.assertLessEqual(legal_action, 170)

    def test_step(self):
        env = rlcard.make('scout')
        state, _ = env.reset()
        action = np.random.choice(list(state['legal_actions'].keys()))
        _, player_id = env.step(action)
        self.assertEqual(player_id, env.game.round.current_player_id)

    # def test_step_back(self):
    #     env = rlcard.make('scout', config={'allow_step_back':False})
    #     state, player_id = env.reset()
    #     action = np.random.choice(list(state['legal_actions'].keys()))
    #     env.step(action)
    #     env.step_back()
    #     self.assertEqual(env.game.round.current_player, player_id)

    #     env = rlcard.make('uno', config={'allow_step_back':False})
    #     state, player_id = env.reset()
    #     action = np.random.choice(list(state['legal_actions'].keys()))
    #     env.step(action)
    #     # env.step_back()
    #     self.assertRaises(Exception, env.step_back)

    def test_run(self):
        env = rlcard.make('scout')
        env.set_agents([RandomAgent(env.num_actions) for _ in range(env.num_players)])
        trajectories, payoffs = env.run(is_training=False)
        self.assertEqual(len(trajectories), 4)

        self.assertEqual(min(payoffs), 0)
        trajectories, payoffs = env.run(is_training=True)
        self.assertGreater(max(payoffs), 0)

    def test_decode_action(self):
        env = rlcard.make('scout')
        env.reset()
        legal_actions = env._get_legal_actions()
        for legal_action in legal_actions:
            decoded: ScoutEvent  = env._decode_action(legal_action)
            self.assertLessEqual(decoded.action_id, legal_action)

    # def test_get_perfect_information(self):
    #     env = rlcard.make('uno')
    #     _, player_id = env.reset()
    #     self.assertEqual(player_id, env.get_perfect_information()['current_player'])
if __name__ == '__main__':
    unittest.main()
