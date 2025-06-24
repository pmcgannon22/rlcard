from collections import Counter, OrderedDict
import numpy as np

from rlcard.envs import Env

from rlcard.games.scout import Game
from rlcard.games.scout.utils import get_action_list
from rlcard.games.scout.utils.action_event import ScoutEvent

ACTION_SPACE = get_action_list()
ACTION_LIST = list(ACTION_SPACE.keys())

DEFAULT_GAME_CONFIG = {
    'game_num_players': 4,
    'hand_size': 16,  # Maximum number of cards in hand
    'allow_step_back': False
}

class ScoutEnv(Env):
    def __init__(self, config: dict):
        self.name = 'scout'
        self.hand_size = config.get('hand_size', DEFAULT_GAME_CONFIG['hand_size'])
        # self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[2, self.hand_size, self.hand_size] for _ in range(self.num_players)]
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
        obs = np.zeros((2, self.hand_size, self.hand_size), dtype=int)  # shape=(2, N=12, R=12)

        # 1) Encode player's hand
        for i, card in enumerate(raw_state['hand']):
            if i >= self.hand_size:
                break  # Limit to hand_size
            rank = card.rank  # must be in range [0..11] or [1..11] etc.
            obs[0, i, rank] = 1

        # 2) Encode table set
        for j, card in enumerate(raw_state['table_set']):
            if j >= self.hand_size:
                break
            rank = card.rank
            obs[1, j, rank] = 1

        # Build up final extracted_state
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
        # legal_ids = self._get_legal_actions()
        # if action_id in legal_ids:
        return ScoutEvent.from_action_id(action_id)
    
    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {action.action_id: None for action in legal_actions}
        return OrderedDict(legal_ids)