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

        # TEMPORAL/HISTORICAL TRACKING
        self.table_history = []  # List of (length, type_priority, max_rank) for last N table sets
        self.max_table_history = 3  # Track last 3 table sets
        self.action_history = {i: [] for i in range(self.num_players)}  # Per-player action history (play=0, scout=1)
        self.action_counts = {i: {'play': 0, 'scout': 0} for i in range(self.num_players)}  # Count actions per player

    def reset(self):
        ''' Reset the environment and clear shaped rewards and history

        Returns:
            (tuple): Tuple containing initial state and player ID
        '''
        self.shaped_rewards = [[] for _ in range(self.num_players)]
        self.prev_state = None

        # Reset temporal tracking
        self.table_history = []
        self.action_history = {i: [] for i in range(self.num_players)}
        self.action_counts = {i: {'play': 0, 'scout': 0} for i in range(self.num_players)}

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

        # UPDATE TEMPORAL/HISTORICAL TRACKING
        from rlcard.games.scout.utils.action_event import PlayAction, ScoutAction

        if isinstance(action, PlayAction):
            # Track that this player played cards
            self.action_counts[current_player_id]['play'] += 1

            # Track table history when a new table set is placed
            table_set = next_state.get('table_set', [])
            if table_set:
                from rlcard.games.scout.utils import segment_strength_rank
                type_priority, max_rank = segment_strength_rank(table_set)
                table_info = (len(table_set), type_priority, max_rank)
                self.table_history.append(table_info)

                # Keep only last N table sets
                if len(self.table_history) > self.max_table_history:
                    self.table_history.pop(0)

        elif isinstance(action, ScoutAction):
            # Track that this player scouted
            self.action_counts[current_player_id]['scout'] += 1

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

        IMPROVEMENTS:
        - Length-scaled combo bonuses
        - Strategic scout rewards
        - Tempo rewards for table control
        - End-game awareness

        Args:
            prev_state (dict): State before action
            action (ScoutEvent): Action taken
            next_state (dict): State after action

        Returns:
            (float): Shaped reward value
        '''
        if not self.use_reward_shaping or prev_state is None:
            return 0.0

        from rlcard.games.scout.utils.action_event import PlayAction, ScoutAction

        reward = 0.0

        # 1. Reward for gaining points (most important signal)
        points_gained = next_state.get('points', 0) - prev_state.get('points', 0)
        reward += points_gained * 2.0

        # 2. Improved scout handling - differentiate strategic vs forced
        hand_size_prev = len(prev_state.get('hand', []))
        hand_size_next = len(next_state.get('hand', []))
        hand_reduction = hand_size_prev - hand_size_next

        is_scout_action = hand_reduction < 0
        was_forced_scout = prev_state.get('current_player_forced_scout', False)

        if is_scout_action:
            if was_forced_scout:
                # Forced scout: small penalty (can't be helped)
                reward -= 0.1
            elif prev_state.get('table_owner') is not None and prev_state.get('table_owner') != self.game.get_player_id():
                # Strategic scout from opponent: this denies them points!
                # In Scout rules, when someone scouts, table owner gets a scout token worth 1 point
                # So scouting from a strong opponent can be strategic
                reward += 0.2  # Small reward for strategic denial
            else:
                # Optional scout from own table or when no good plays
                reward -= 0.15

        # 3. Length-scaled combo rewards (playing cards)
        elif hand_reduction > 0:
            # Base reward for reducing hand
            reward += hand_reduction * 0.05

            # Length-scaled bonus (exponential scaling for long combos)
            if hand_reduction == 1:
                reward += 0.05  # Single card
            elif hand_reduction == 2:
                reward += 0.15  # Pair/short run
            elif hand_reduction == 3:
                reward += 0.30  # Triple/medium run
            elif hand_reduction >= 4:
                # Big combo: substantial reward scaling with length
                reward += 0.50 + 0.10 * (hand_reduction - 4)

        # 4. Tempo rewards: table control is valuable
        prev_owner = prev_state.get('table_owner')
        next_owner = next_state.get('table_owner')
        my_id = self.game.get_player_id()

        if isinstance(action, PlayAction):
            # Gained table control
            if prev_owner != my_id and next_owner == my_id:
                reward += 0.3
            # Maintained table control
            elif prev_owner == my_id and next_owner == my_id:
                reward += 0.1
            # Lost table control (shouldn't happen with play, but just in case)
            elif prev_owner == my_id and next_owner != my_id:
                reward -= 0.2

        # 5. End-game awareness: strategy changes when close to ending
        avg_hand_size = sum(next_state.get('num_cards', {}).values()) / max(1, len(next_state.get('num_cards', {})))
        is_late_game = avg_hand_size < 5

        if is_late_game:
            # Late game: check if we're winning or losing
            my_score = next_state.get('points', 0)
            opponent_scores = [next_state.get('num_cards', {}).get(i, 0) for i in range(self.num_players) if i != my_id]

            # In late game, if winning, defensive play (maintaining lead) is good
            if opponent_scores and my_score > max(opponent_scores):
                # Winning: reward maintaining table control and not giving up points
                if next_owner == my_id:
                    reward += 0.15
            else:
                # Losing: reward aggressive play (reducing hand size fast)
                if hand_reduction > 0:
                    reward += 0.10 * hand_reduction

        # 6. Potential-based shaping for game state quality
        gamma = 0.99
        prev_potential = self._compute_state_potential(prev_state)
        next_potential = self._compute_state_potential(next_state)
        reward += gamma * next_potential - prev_potential

        return reward

    def _compute_state_potential(self, state):
        ''' Compute potential function for state (higher is better)

        This provides a heuristic measure of how good a game state is.

        IMPROVEMENTS:
        - Relative standing vs opponents
        - Win progress measure
        - Tempo advantage

        Args:
            state (dict): Game state

        Returns:
            (float): Potential value
        '''
        potential = 0.0

        # Current score is valuable
        my_score = state.get('points', 0)
        potential += my_score * 1.0

        # Fewer cards in hand is better (closer to winning)
        hand_size = len(state.get('hand', []))
        potential -= hand_size * 0.15

        # Being table owner is advantageous (tempo)
        my_id = self.game.get_player_id()
        if state.get('table_owner') == my_id:
            potential += 0.5

        # Relative standing: compare to opponents
        num_cards_dict = state.get('num_cards', {})
        if num_cards_dict:
            # Better if opponents have more cards
            opponent_avg_cards = sum(num_cards_dict.get(i, 0) for i in range(self.num_players) if i != my_id) / max(1, self.num_players - 1)
            card_advantage = opponent_avg_cards - hand_size
            potential += card_advantage * 0.1

        # Win progress: how close to ending the game
        # If we have very few cards, that's good progress
        if hand_size <= 3:
            potential += (3 - hand_size) * 0.3

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
        from rlcard.games.scout.utils.action_event import PlayAction, ScoutAction

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
        # table front/back rank hints, WIN CONDITION AWARENESS features:
        # game_phase, my_ranking, cards_until_end, winning_margin,
        # RICH TABLE SET ENCODING features:
        # is_single, is_run, is_group, type_priority, min_rank_to_beat, playable_segment_count,
        # TEMPORAL/HISTORICAL features:
        # last_3_tables (3 * 3 = 9), scout_ratio_per_player (4),
        # and ENHANCED OPPONENT MODELING features:
        # all_player_scores (4)
        self._extra_scalar_features = 31  # Increased from 27 to 31 (added 4 opponent score features)
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
            + self.hand_size * 2       # combo features: combo_mask + combo_strength
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
        # EXPLICIT COMBO FEATURES
        self._obs_slices['combo_mask'] = (start, start + self.hand_size)
        start += self.hand_size
        self._obs_slices['combo_strength'] = (start, start + self.hand_size)
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

        # EXPLICIT COMBO FEATURES: compute which cards are in combos and their strength
        combo_mask, combo_strength = self._compute_combo_features(raw_state['hand'])

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

        # WIN CONDITION AWARENESS FEATURES
        # 1. Game phase: early (0.0), mid (0.5), late (1.0) based on average hand size
        avg_hand_size = sum(num_cards.values()) / max(1, len(num_cards))
        if avg_hand_size > 10:
            game_phase = 0.0  # Early game
        elif avg_hand_size > 5:
            game_phase = 0.5  # Mid game
        else:
            game_phase = 1.0  # Late game
        info_vec[offset] = game_phase
        offset += 1

        # 2. My ranking: normalized position (1.0 = first place, 0.0 = last place)
        # Compare based on scores (higher is better in Scout)
        my_id = self.game.get_player_id()
        my_score = raw_state.get('points', 0)
        opponent_scores = [num_cards.get(i, 0) for i in range(self.num_players) if i != my_id]
        # Count how many opponents have lower scores
        better_than_count = sum(1 for score in opponent_scores if my_score > score)
        my_ranking = better_than_count / max(1, self.num_players - 1)
        info_vec[offset] = my_ranking
        offset += 1

        # 3. Cards until end: minimum hand size across all players (normalized)
        min_hand_size = min(num_cards.values()) if num_cards else self.hand_size
        info_vec[offset] = min_hand_size / max(1, self.hand_size)
        offset += 1

        # 4. Winning margin: my score - max opponent score (normalized by max possible score)
        max_opponent_score = max(opponent_scores) if opponent_scores else 0
        winning_margin = (my_score - max_opponent_score) / max(1, self.hand_size)
        # Clamp to [-1, 1] range
        winning_margin = max(-1.0, min(1.0, winning_margin))
        info_vec[offset] = winning_margin
        offset += 1

        # RICH TABLE SET ENCODING FEATURES
        table_set = raw_state.get('table_set', [])
        hand = raw_state.get('hand', [])

        # Analyze table set type and strength
        is_single, is_run, is_group, type_priority, min_rank_to_beat = self._analyze_table_set(table_set)

        info_vec[offset] = is_single
        offset += 1
        info_vec[offset] = is_run
        offset += 1
        info_vec[offset] = is_group
        offset += 1
        info_vec[offset] = type_priority
        offset += 1
        info_vec[offset] = min_rank_to_beat
        offset += 1

        # Count how many segments in my hand can beat this table
        playable_count = self._count_playable_segments(hand, table_set)
        # Normalize by max possible segments (rough estimate: hand_size / 2)
        max_possible_segments = max(1, self.hand_size // 2)
        playable_count_norm = min(1.0, playable_count / max_possible_segments)
        info_vec[offset] = playable_count_norm
        offset += 1

        # TEMPORAL/HISTORICAL FEATURES
        # Last 3 table sets played: (length, type_priority, max_rank) for each
        for i in range(self.max_table_history):
            if i < len(self.table_history):
                table_length, type_priority, max_rank = self.table_history[-(i+1)]  # Most recent first
                info_vec[offset] = table_length / max(1, self.hand_size)
                offset += 1
                info_vec[offset] = type_priority / 2.0  # Normalize (0, 1, 2) -> [0, 1]
                offset += 1
                info_vec[offset] = self._normalize_rank_value(max_rank)
                offset += 1
            else:
                # No table in this history slot
                info_vec[offset] = 0.0
                offset += 1
                info_vec[offset] = 0.0
                offset += 1
                info_vec[offset] = 0.0
                offset += 1

        # Scout ratio per player: scouts / (scouts + plays)
        for pid in range(self.num_players):
            play_count = self.action_counts[pid]['play']
            scout_count = self.action_counts[pid]['scout']
            total_actions = play_count + scout_count
            scout_ratio = scout_count / max(1, total_actions)
            info_vec[offset] = scout_ratio
            offset += 1

        # ENHANCED OPPONENT MODELING: All player scores
        all_scores = self._get_all_player_scores()
        for pid in range(self.num_players):
            score = all_scores.get(pid, 0)
            # Normalize by max possible score (rough estimate: hand_size)
            normalized_score = score / max(1, self.hand_size)
            info_vec[offset] = normalized_score
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
            combo_mask,          # NEW: Which cards are in combos
            combo_strength,      # NEW: Strength of combos for each card
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

    def _get_all_player_scores(self):
        """Get scores for all players.

        Returns:
            dict: {player_id: score} for all players
        """
        try:
            if hasattr(self.game, 'round') and self.game.round is not None:
                return {i: self.game.round.players[i].score for i in range(self.num_players)}
        except:
            pass
        # Fallback: return zeros
        return {i: 0 for i in range(self.num_players)}

    def _analyze_table_set(self, table_set):
        """Analyze the table set to determine its type and strength.

        Returns:
            tuple: (is_single, is_run, is_group, type_priority, min_rank_to_beat)
        """
        if not table_set:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        from rlcard.games.scout.utils import segment_strength_rank

        length = len(table_set)

        # Determine type
        is_single = 1.0 if length == 1 else 0.0

        # Check if it's a group (all same rank)
        is_group = 1.0 if all(c.rank == table_set[0].rank for c in table_set) else 0.0

        # Check if it's a run (consecutive ranks)
        is_run = 0.0
        if not is_group and not is_single:
            is_ascending = all(table_set[i+1].rank - table_set[i].rank == 1
                             for i in range(length - 1))
            is_descending = all(table_set[i].rank - table_set[i+1].rank == 1
                               for i in range(length - 1))
            if is_ascending or is_descending:
                is_run = 1.0

        # Get strength using existing utility
        type_priority, max_rank = segment_strength_rank(table_set)

        # Normalize type_priority (0, 1, or 2) to [0, 1]
        type_priority_norm = type_priority / 2.0

        # Min rank to beat depends on type
        # For same-length combos: need higher rank OR better type
        min_rank_to_beat = self._normalize_rank_value(max_rank + 1)

        return (is_single, is_run, is_group, type_priority_norm, min_rank_to_beat)

    def _count_playable_segments(self, hand, table_set):
        """Count how many segments in hand can beat the current table set.

        Returns:
            int: Number of playable segments that beat the table
        """
        if not table_set:
            # If table is empty, all segments are playable
            from rlcard.games.scout.utils import find_all_scout_segments
            segments = find_all_scout_segments(hand)
            return len(segments)

        from rlcard.games.scout.utils import find_all_scout_segments, segment_strength_rank

        segments = find_all_scout_segments(hand)
        playable_count = 0

        # Check each segment against table set
        for segment in segments:
            cards = segment['cards']
            # Use game logic to check if this beats table
            # Simplified: check length and strength
            if len(cards) > len(table_set):
                playable_count += 1
            elif len(cards) == len(table_set):
                new_strength = segment_strength_rank(cards)
                old_strength = segment_strength_rank(table_set)

                # Same type, need higher rank
                if new_strength[0] > old_strength[0]:
                    playable_count += 1
                elif new_strength[0] == old_strength[0] and new_strength[1] > old_strength[1]:
                    playable_count += 1

        return playable_count

    def _compute_combo_features(self, hand):
        """Compute explicit combo features for each card in hand.

        For each card position, compute:
        - is_in_combo: whether this card is part of any playable combo
        - combo_strength: normalized strength of the best combo including this card

        Returns:
            tuple: (combo_mask, combo_strength) where each is array of length hand_size
        """
        combo_mask = np.zeros(self.hand_size, dtype=np.float32)
        combo_strength = np.zeros(self.hand_size, dtype=np.float32)

        if not hand:
            return combo_mask, combo_strength

        from rlcard.games.scout.utils import find_all_scout_segments, segment_strength_rank

        # Find all valid segments in hand
        segments = find_all_scout_segments(hand)

        # For each card position, track if it's in any combo and its max strength
        for segment in segments:
            start_idx = segment['start']
            end_idx = segment['end']
            cards = segment['cards']

            # Get strength of this combo
            type_priority, max_rank = segment_strength_rank(cards)
            # Normalize: combine type (0-2) and rank (1-10) into a single strength score
            # Strength = type_priority * 10 + max_rank, then normalize to [0, 1]
            # Max possible: 2 * 10 + 10 = 30
            strength = (type_priority * 10 + max_rank) / 30.0

            # Mark all cards in this segment
            for i in range(start_idx, end_idx):
                if i < self.hand_size:
                    combo_mask[i] = 1.0
                    # Keep the maximum strength for each position
                    combo_strength[i] = max(combo_strength[i], strength)

        return combo_mask, combo_strength
