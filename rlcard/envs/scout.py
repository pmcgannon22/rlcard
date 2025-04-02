from collections import Counter, OrderedDict
import numpy as np

from rlcard.envs import Env

from rlcard.games.scout import Game
from rlcard.games.scout.utils import get_action_list
from rlcard.games.scout.utils.action_event import ScoutEvent

ACTION_SPACE = get_action_list()
ACTION_LIST = list(ACTION_SPACE.keys())

class ScoutEnv(Env):
    def __init__(self, config):
        self.name = 'scout'
        # self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[2, 12, 11] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, raw_state):
        """
        raw_state: dict with keys like:
        {
            'hand': list of ScoutCard (already oriented),
            'table_set': list of ScoutCard (or empty),
            # ...
        }
        """
        obs = np.zeros((2, 12, 12), dtype=int)  # shape=(2, N=12, R=12)

        # 1) Encode player's hand
        for i, card in enumerate(raw_state['hand']):
            if i >= 12:
                break  # we only have space for 12
            rank = card.rank  # must be in range [0..11] or [1..11] etc.
            obs[0, i, rank] = 1

        # 2) Encode table set
        for j, card in enumerate(raw_state['table_set']):
            if j >= 12:
                break
            rank = card.rank
            obs[1, j, rank] = 1

        # Build up final extracted_state
        extracted_state = {
            'obs': obs,
            'legal_actions': self._get_legal_actions(),
            'raw_obs': raw_state,
            'raw_legal_actions': raw_state['legal_actions'],  # etc.
        }

        return extracted_state

    def get_payoffs(self):
         return self.game.get_payoffs()

    def _decode_action(self, action_id):
        # legal_ids = self._get_legal_actions()
        # if action_id in legal_ids:
        return ScoutEvent.from_action_id(action_id)
    
    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {action.action_id: None for action in legal_actions}
        return OrderedDict(legal_ids)