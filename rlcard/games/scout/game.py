from copy import deepcopy
import numpy as np

from rlcard.games.scout import Dealer
from rlcard.games.scout import Player
from rlcard.games.scout import Round
from rlcard.games.scout.utils.action_event import ScoutEvent

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
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_players, his_round))

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
        return 170

    def get_player_id(self):
        ''' Return the current player that will take actions soon
        '''
        return self.round.current_player_id
    
    def is_over(self):
        ''' Return whether the current game is over
        '''
        return self.round.game_over
    
    def get_current_player(self):
        return self.round.get_current_player()
    
    def get_legal_actions(self) -> list[ScoutEvent]:
        return self.round.get_legal_actions()
    
    def get_state(self, player_id):
        return self.round.get_state(player_id)