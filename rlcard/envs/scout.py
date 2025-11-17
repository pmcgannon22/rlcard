from collections import OrderedDict
from typing import Dict, Tuple, Optional

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
        self.game = Game(max_hand_size=self.hand_size)
        super().__init__(config)
        self._build_observation_spec()
        self.state_shape = [[self._obs_vector_length] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

        # Reward shaping configuration
        self.use_reward_shaping = config.get('reward_shaping', True)
        self.prev_state = None  # Track previous state for reward shaping
        self.shaped_rewards = [[] for _ in range(self.num_players)]  # Track shaped rewards per player

    def reset(self):
        ''' Reset the environment and clear shaped rewards

        Returns:
            (tuple): Tuple containing initial state and player ID
        '''
        self.shaped_rewards = [[] for _ in range(self.num_players)]
        self.prev_state = None
        return super().reset()

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

        # Get current state before the action for context and reward shaping
        current_player_id = self.get_player_id()
        current_state = self.game.get_state(current_player_id)

        # Record the action with enhanced context
        action_context = self._get_action_context(action, current_state)
        self.action_recorder.append((current_player_id, action, action_context))

        # Execute the game step
        next_state, player_id = self.game.step(action)

        # Compute and store shaped reward for this transition
        if self.use_reward_shaping:
            shaped_reward = self._compute_shaped_reward(current_state, action, next_state)
            self.shaped_rewards[current_player_id].append(shaped_reward)

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
            context['start_idx'] = action.start_idx
            context['end_idx'] = action.end_idx
            context['action_id'] = action.action_id
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
            context['insertion'] = action.insertion_in_hand
            context['action_type'] = 'scout'
            context['action_id'] = action.action_id
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

    def _compute_shaped_reward(self, prev_state, action, next_state):
        ''' Compute shaped reward to provide intermediate feedback

        Reward shaping provides immediate feedback on actions to accelerate learning.
        Uses potential-based shaping to maintain optimal policy guarantees.

        Args:
            prev_state (dict): State before action
            action (ScoutEvent): Action taken
            next_state (dict): State after action

        Returns:
            (float): Shaped reward value
        '''
        if not self.use_reward_shaping or prev_state is None:
            return 0.0

        reward = 0.0

        # 1. Reward for gaining points (most important signal)
        points_gained = next_state.get('points', 0) - prev_state.get('points', 0)
        reward += points_gained * 2.0

        # 2. Penalty for forced scout (couldn't play any cards)
        if next_state.get('current_player_forced_scout', False):
            reward -= 0.2

        # 3. Small penalty for hand size (encourage playing cards)
        hand_size_prev = len(prev_state.get('hand', []))
        hand_size_next = len(next_state.get('hand', []))
        hand_reduction = hand_size_prev - hand_size_next
        if hand_reduction > 0:
            # Reward for reducing hand size
            reward += hand_reduction * 0.1
            # Bonus for playing long combos
            if hand_reduction >= 4:
                reward += 0.3
        elif hand_reduction < 0:
            # Small penalty for scouting (increasing hand size)
            reward -= 0.05

        # 4. Potential-based shaping for game state quality
        gamma = 0.99
        prev_potential = self._compute_state_potential(prev_state)
        next_potential = self._compute_state_potential(next_state)
        reward += gamma * next_potential - prev_potential

        return reward

    def _compute_state_potential(self, state):
        ''' Compute potential function for state (higher is better)

        This provides a heuristic measure of how good a game state is.

        Args:
            state (dict): Game state

        Returns:
            (float): Potential value
        '''
        potential = 0.0

        # Current score is valuable
        potential += state.get('points', 0) * 1.0

        # Fewer cards in hand is better (closer to winning)
        hand_size = len(state.get('hand', []))
        potential -= hand_size * 0.2

        # Being table owner is advantageous
        if state.get('table_owner') == self.game.get_player_id():
            potential += 0.5

        return potential

    def _decode_action(self, action_id):
        # Generate the action list for the current player's hand size
        action_list = get_action_list(self.hand_size)
        return ScoutEvent.from_action_id(action_id, action_list)
    
    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {action.action_id: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def get_action_feature(self, action_id):
        ''' Get semantic features for an action

        This method encodes actions with meaningful features that help the network
        understand action semantics, rather than treating action IDs as arbitrary integers.

        Args:
            action_id (int): The ID of the action

        Returns:
            (np.ndarray): A feature vector representing the action semantics
        '''
        from .utils.action_event import PlayAction, ScoutAction

        # Decode the action
        action = self._decode_action(action_id)

        # Get current game state to extract card information
        try:
            state = self.game.get_state(self.game.get_player_id())
            hand = state.get('hand', [])
        except:
            hand = []

        if isinstance(action, PlayAction):
            # Extract segment information
            start_idx = action.start_idx
            end_idx = action.end_idx
            combo_length = end_idx - start_idx

            # Get cards if available
            if hand and end_idx <= len(hand):
                segment = hand[start_idx:end_idx]
                min_rank = min([c.rank for c in segment])
                max_rank = max([c.rank for c in segment])

                # Determine combo type
                if combo_length == 1:
                    combo_type = 0  # Single
                elif all(c.rank == segment[0].rank for c in segment):
                    combo_type = 2  # Group (same rank)
                else:
                    combo_type = 1  # Run (consecutive)
            else:
                min_rank = 0
                max_rank = 0
                combo_type = 0

            return np.array([
                0.0,                              # Action type: play
                start_idx / float(self.hand_size),  # Normalized position
                end_idx / float(self.hand_size),    # Normalized end position
                combo_length / float(self.hand_size),  # Normalized length
                combo_type / 2.0,                   # Combo type (0=single, 1=run, 2=group)
                min_rank / float(self.rank_count),  # Min rank in combo
                max_rank / float(self.rank_count),  # Max rank in combo
            ], dtype=np.float32)

        elif isinstance(action, ScoutAction):
            return np.array([
                1.0,                                          # Action type: scout
                1.0 if action.from_front else 0.0,            # Front or back
                action.insertion_in_hand / float(self.hand_size),  # Insert position
                1.0 if action.flip else 0.0,                  # Flipped
                0.0, 0.0, 0.0                                 # Padding to match play features
            ], dtype=np.float32)

        else:
            # Unknown action type - return zeros
            return np.zeros(7, dtype=np.float32)

    def _build_observation_spec(self) -> None:
        """Pre-compute observation layout so ordering is preserved."""
        # DENSE ENCODING: Use normalized values instead of one-hot
        # Each card position has 2 features: normalized top and bottom values
        self._card_plane_size = self.hand_size * 2  # Changed from hand_size * rank_count
        # owner one-hot (+1 for None), consecutive scouts, per-player hand counts,
        # player score, table length, orientation flag, forced-scout indicator,
        # and table front/back rank hints
        self._extra_scalar_features = 4
        self._info_vector_len = (
            (self.num_players + 1)
            + 1
            + self.num_players
            + 2
            + self._extra_scalar_features
        )
        self._obs_vector_length = (
            self._card_plane_size * 2  # hand and table (primary & secondary are now combined)
            + self.hand_size * 2       # hand/table occupancy masks
            + self._info_vector_len
        )
        start = 0
        self._obs_slices: Dict[str, Tuple[int, int]] = {}
        # Hand cards: hand_size * 2 (top, bottom for each position)
        self._obs_slices['hand_cards'] = (start, start + self._card_plane_size)
        start += self._card_plane_size
        # Table cards: hand_size * 2 (top, bottom for each position)
        self._obs_slices['table_cards'] = (start, start + self._card_plane_size)
        start += self._card_plane_size
        self._obs_slices['hand_mask'] = (start, start + self.hand_size)
        start += self.hand_size
        self._obs_slices['table_mask'] = (start, start + self.hand_size)
        start += self.hand_size
        self._obs_slices['info'] = (start, start + self._info_vector_len)

    def _build_observation_vector(self, raw_state: dict) -> np.ndarray:
        """Encode ordered hand/table information plus scalar context.

        DENSE ENCODING: Each card is represented by 2 normalized values (top, bottom)
        instead of one-hot encoding. This reduces dimensionality and preserves
        ordinal relationships between ranks.
        """
        # Hand cards: shape (hand_size, 2) for [top_value, bottom_value]
        hand_cards = np.zeros((self.hand_size, 2), dtype=np.float32)
        # Table cards: shape (hand_size, 2) for [top_value, bottom_value]
        table_cards = np.zeros((self.hand_size, 2), dtype=np.float32)
        hand_mask = np.zeros(self.hand_size, dtype=np.float32)
        table_mask = np.zeros(self.hand_size, dtype=np.float32)

        # Encode hand cards with normalized dense values
        for i, card in enumerate(raw_state['hand']):
            if i >= self.hand_size:
                break
            hand_cards[i, 0] = self._normalize_rank_value(card.rank)
            hand_cards[i, 1] = self._normalize_rank_value(card.bottom)
            hand_mask[i] = 1.0

        # Encode table cards with normalized dense values
        for j, card in enumerate(raw_state['table_set']):
            if j >= self.hand_size:
                break
            table_cards[j, 0] = self._normalize_rank_value(card.rank)
            table_cards[j, 1] = self._normalize_rank_value(card.bottom)
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
        offset += 1

        # Orientation (1 if flipped), forced scout flag, and table front/back ranks
        info_vec[offset] = 1.0 if raw_state.get('orientation_flipped') else 0.0
        offset += 1
        info_vec[offset] = 1.0 if raw_state.get('current_player_forced_scout') else 0.0
        offset += 1
        info_vec[offset] = self._normalize_rank_value(raw_state.get('table_front_top', 0))
        offset += 1
        info_vec[offset] = self._normalize_rank_value(raw_state.get('table_back_top', 0))
        offset += 1

        assert (
            offset == self._info_vector_len
        ), f"Scout info vector mis-sized ({offset} != {self._info_vector_len})"

        # Concatenate all observation components
        # DENSE ENCODING: hand_cards and table_cards are now (hand_size, 2) arrays
        obs_components = [
            hand_cards.ravel(),   # Flatten (hand_size, 2) -> (hand_size * 2,)
            table_cards.ravel(),  # Flatten (hand_size, 2) -> (hand_size * 2,)
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

    def _normalize_rank_value(self, rank_value: Optional[int]) -> float:
        """Map public rank info into [0,1] for scalar channels."""
        if not rank_value:
            return 0.0
        clamped = max(1, min(rank_value, self.rank_count))
        return clamped / float(self.rank_count)
