from copy import deepcopy
import numpy as np
from typing import List
from math import ceil

from rlcard.games.scout import Dealer
from rlcard.games.scout import Player
from rlcard.games.scout import Round
from rlcard.games.scout.utils.action_event import ScoutEvent
from rlcard.games.scout.utils import get_action_list

class ScoutGame:
    def __init__(self, num_players=4, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        # self.payoffs = [0 for _ in range(self.num_players)]
    
    # def configure(self, game_config):
    #     ''' Specifiy some game specific parameters, such as number of players
    #     '''
    #     self.num_players = game_config['game_num_players']

    def init_game(self):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id
        '''
        # Initalize payoffs
        # self.payoffs = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize a Round
        self.round = Round(self.dealer, self.num_players)

        # Save the hisory for stepping back to the last state.
        self.history = []

        player_id = self.round.current_player_id
        state = self.round.get_state(player_id)
        return state, player_id
    
    def step(self, action):
        ''' Get the next state

        Args:
            action (ScoutEvent): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next player's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            self.history.append((his_dealer, his_round))

        player_id, _ = self.round.proceed_round(action)
        state = self.round.get_state(player_id)
        return state, player_id
    
    def get_payoffs(self):
        return self.round.get_payoffs()

    def get_num_players(self):
        ''' Return the number of players in the game
        '''
        return self.num_players
    
    def get_num_actions(self):
        # If round is initialized, use actual hand sizes
        if hasattr(self, 'round') and hasattr(self.round, 'players'):
            max_hand_size = max(len(player.hand) for player in self.round.players)
        else:
            # Estimate max hand size before game is initialized
            # There are 45 cards in the deck, divided among num_players
            max_hand_size = ceil(45 / self.num_players)
        return len(get_action_list(max_hand_size))

    def get_player_id(self):
        ''' Return the current player that will take actions soon
        '''
        return self.round.current_player_id
    
    def is_over(self):
        ''' Return whether the current game is over
        '''
        return self.round.game_over
    
    def get_legal_actions(self) -> List[ScoutEvent]:
        return self.round.get_legal_actions()
    
    def get_state(self, player_id):
        return self.round.get_state(player_id)
    
    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['num_players'] = self.num_players
        state['hand_cards'] = [player.hand for player in self.round.players]
        state['table_set'] = self.round.table_set
        state['table_owner'] = self.round.table_owner
        state['current_player'] = self.round.current_player_id
        state['legal_actions'] = self.round.get_legal_actions()
        state['consecutive_scouts'] = self.round.consecutive_scouts
        return state