from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np

from rlcard.envs import Env
from rlcard.games.scout import Game
from rlcard.games.scout.utils import get_action_list
from rlcard.games.scout.utils.action_event import ScoutEvent

ACTION_SPACE = get_action_list(max_hand_size=16)
ACTION_LIST = list(ACTION_SPACE.keys())

DEFAULT_GAME_CONFIG = {
    'game_num_players': 4,
    'hand_size': 16,  # Maximum number of cards in hand
    'allow_step_back': False,
    'rank_count': 10,
}

class ScoutEnv(Env):
    def __init__(self, config: dict):
        self.name = 'scout'
        self.hand_size = config.get('hand_size', DEFAULT_GAME_CONFIG['hand_size'])
        self.rank_count = config.get('rank_count', DEFAULT_GAME_CONFIG['rank_count'])
        self.game = Game()
        super().__init__(config)
        self._build_observation_spec()
        self.state_shape = [[self._obs_vector_length] for _ in range(self.num_players)]
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
        obs = self._build_observation_vector(raw_state)

        extracted_state = {
            'obs': obs,
            'legal_actions': self._get_legal_actions(),
            'raw_obs': raw_state,
            'raw_legal_actions': raw_state['legal_actions'],  # etc.
            'action_record': self.action_recorder,
        }

        return extracted_state

    def step(self, action, raw_action=False):
        ''' Step forward with enhanced action recording

        Args:
            action (int): The action taken by the current player
            raw_action (boolean): True if the action is a raw action

        Returns:
            (tuple): Tuple containing:

                (dict): The next state
                (int): The ID of the next player
        '''
        if not raw_action:
            action = self._decode_action(action)

        self.timestep += 1
        
        # Get current state before the action for context
        current_state = self.game.get_state(self.get_player_id())
        
        # Record the action with enhanced context
        action_context = self._get_action_context(action, current_state)
        self.action_recorder.append((self.get_player_id(), action, action_context))
        
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def _get_action_context(self, action, state):
        ''' Get context information about an action for enhanced display

        Args:
            action (ScoutEvent): The action being taken
            state (dict): The current game state

        Returns:
            (dict): Context information about the action
        '''
        context = {}
        
        if hasattr(action, 'start_idx') and hasattr(action, 'end_idx'):  # PlayAction
            # Get the cards being played
            cards = state['hand'][action.start_idx:action.end_idx]
            context['cards'] = [f'{card.top}/{card.bottom}' for card in cards]
            context['action_type'] = 'play'
        elif hasattr(action, 'from_front') and hasattr(action, 'insertion_in_hand'):  # ScoutAction
            # Get the card being scouted
            if state['table_set']:
                if action.from_front:
                    scout_card = state['table_set'][0]
                else:
                    scout_card = state['table_set'][-1]
                context['card'] = f'{scout_card.top}/{scout_card.bottom}'
                context['direction'] = 'front' if action.from_front else 'back'
                context['flipped'] = action.flip if hasattr(action, 'flip') else False
            context['action_type'] = 'scout'
        
        return context

    def get_payoffs(self):
         return np.array(self.game.get_payoffs())

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        return self.game.get_perfect_information()

    def _decode_action(self, action_id):
        # Generate the action list for the current player's hand size
        player = self.game.round.players[self.game.round.current_player_id]
        action_list = get_action_list(len(player.hand))
        return ScoutEvent.from_action_id(action_id, action_list)
    
    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {action.action_id: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def _build_observation_spec(self) -> None:
        """Pre-compute observation layout so ordering is preserved."""
        self._card_plane_size = self.hand_size * self.rank_count
        # owner one-hot (+1 for None), consecutive scouts, per-player hand counts,
        # player score, table length
        self._info_vector_len = (self.num_players + 1) + 1 + self.num_players + 2
        self._obs_vector_length = (
            self._card_plane_size * 4  # hand/table primary & secondary values
            + self.hand_size * 2       # hand/table occupancy masks
            + self._info_vector_len
        )
        start = 0
        self._obs_slices: Dict[str, Tuple[int, int]] = {}
        for key in ('hand_primary', 'hand_secondary', 'table_primary', 'table_secondary'):
            end = start + self._card_plane_size
            self._obs_slices[key] = (start, end)
            start = end
        self._obs_slices['hand_mask'] = (start, start + self.hand_size)
        start += self.hand_size
        self._obs_slices['table_mask'] = (start, start + self.hand_size)
        start += self.hand_size
        self._obs_slices['info'] = (start, start + self._info_vector_len)

    def _build_observation_vector(self, raw_state: dict) -> np.ndarray:
        """Encode ordered hand/table information plus scalar context."""
        hand_primary = np.zeros((self.hand_size, self.rank_count), dtype=np.float32)
        hand_secondary = np.zeros_like(hand_primary)
        table_primary = np.zeros_like(hand_primary)
        table_secondary = np.zeros_like(hand_primary)
        hand_mask = np.zeros(self.hand_size, dtype=np.float32)
        table_mask = np.zeros(self.hand_size, dtype=np.float32)

        for i, card in enumerate(raw_state['hand']):
            if i >= self.hand_size:
                break
            hand_primary[i, self._rank_to_index(card.rank)] = 1.0
            hand_secondary[i, self._rank_to_index(card.bottom)] = 1.0
            hand_mask[i] = 1.0

        for j, card in enumerate(raw_state['table_set']):
            if j >= self.hand_size:
                break
            table_primary[j, self._rank_to_index(card.rank)] = 1.0
            table_secondary[j, self._rank_to_index(card.bottom)] = 1.0
            table_mask[j] = 1.0

        info_vec = np.zeros(self._info_vector_len, dtype=np.float32)
        owner_one_hot_len = self.num_players + 1
        owner_idx = raw_state.get('table_owner')
        owner_position = owner_idx if owner_idx is not None else self.num_players
        info_vec[owner_position] = 1.0
        offset = owner_one_hot_len
        consecutive = raw_state.get('consecutive_scouts', 0)
        info_vec[offset] = consecutive / max(1, self.num_players - 1)
        offset += 1
        num_cards = raw_state.get('num_cards', {})
        for pid in range(self.num_players):
            info_vec[offset + pid] = num_cards.get(pid, 0) / max(1, self.hand_size)
        offset += self.num_players
        info_vec[offset] = raw_state.get('points', 0) / max(1, self.hand_size)
        offset += 1
        info_vec[offset] = len(raw_state.get('table_set', [])) / max(1, self.hand_size)

        obs_components = [
            hand_primary.ravel(),
            hand_secondary.ravel(),
            table_primary.ravel(),
            table_secondary.ravel(),
            hand_mask,
            table_mask,
            info_vec,
        ]
        return np.concatenate(obs_components).astype(np.float32, copy=False)

    def _rank_to_index(self, rank_value: int) -> int:
        """Clamp rank into valid index range."""
        if rank_value is None:
            return 0
        return max(1, min(rank_value, self.rank_count)) - 1
