from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from rlcard.games.scout import Dealer
from rlcard.games.scout import Round
from rlcard.games.scout.utils.action_event import ScoutEvent
from rlcard.games.scout.utils import get_action_list


class ScoutGame:
    def __init__(
        self,
        num_players: int = 4,
        allow_step_back: bool = False,
        force_play_if_possible: bool = True,
        max_hand_size: int = 16,
    ) -> None:
        self.allow_step_back: bool = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players: int = num_players
        self.force_play_if_possible: bool = force_play_if_possible
        self.max_hand_size: int = max_hand_size
        self.dealer: Dealer | None = None
        self.round: Round | None = None
        self.history: list[tuple[Dealer, Round]] = []
        # self.payoffs = [0 for _ in range(self.num_players)]
    
    # def configure(self, game_config):
    #     ''' Specifiy some game specific parameters, such as number of players
    #     '''
    #     self.num_players = game_config['game_num_players']

    def init_game(self) -> tuple[dict[str, Any], int]:
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
        self.round = Round(
            self.dealer,
            self.num_players,
            self.force_play_if_possible,
            self.max_hand_size,
        )

        # Save the hisory for stepping back to the last state.
        self.history = []

        player_id = self.round.current_player_id
        state = self.round.get_state(player_id)
        return state, player_id
    
    def step(self, action: ScoutEvent) -> tuple[dict[str, Any], int]:
        ''' Get the next state

        Args:
            action (ScoutEvent): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next player's id
        '''

        if self.round is None or self.dealer is None:
            raise RuntimeError("Game has not been initialized. Call init_game() first.")

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            self.history.append((his_dealer, his_round))

        player_id, _ = self.round.proceed_round(action)
        state = self.round.get_state(player_id)
        return state, player_id
    
    def get_payoffs(self) -> list[int]:
        return self._require_round().get_payoffs()

    def get_num_players(self) -> int:
        ''' Return the number of players in the game
        '''
        return self.num_players
    
    def get_num_actions(self) -> int:
        # If round is initialized, use actual hand sizes
        # if hasattr(self, 'round') and hasattr(self.round, 'players'):
        #     max_hand_size = max(len(player.hand) for player in self.round.players)
        # else:
        #     # Estimate max hand size before game is initialized
        #     # There are 45 cards in the deck, divided among num_players
        #     max_hand_size = ceil(45 / self.num_players)
        return len(get_action_list())

    def get_player_id(self) -> int:
        ''' Return the current player that will take actions soon
        '''
        return self._require_round().current_player_id
    
    def is_over(self) -> bool:
        ''' Return whether the current game is over
        '''
        return self._require_round().game_over
    
    def get_legal_actions(self) -> list[ScoutEvent]:
        return self._require_round().get_legal_actions()
    
    def get_state(self, player_id: int) -> dict[str, Any]:
        return self._require_round().get_state(player_id)
    
    def get_perfect_information(self) -> dict[str, Any]:
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        round_obj = self._require_round()
        state: dict[str, Any] = {}
        state['num_players'] = self.num_players
        state['hand_cards'] = [player.hand for player in round_obj.players]
        state['table_set'] = round_obj.table_set
        state['table_owner'] = round_obj.table_owner
        state['current_player'] = round_obj.current_player_id
        state['legal_actions'] = round_obj.get_legal_actions()
        state['consecutive_scouts'] = round_obj.consecutive_scouts
        return state

    def _require_round(self) -> Round:
        if self.round is None:
            raise RuntimeError("Round has not been initialized. Call init_game() first.")
        return self.round

    def set_orientation(self, player_id: int, reverse: bool = False) -> None:
        self._require_round().choose_orientation(player_id, reverse)
