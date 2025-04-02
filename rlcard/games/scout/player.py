from enum import Enum

from rlcard.games.scout.card import ScoutCard

class PlayerStatus(Enum):
    ALIVE = 0
    OUT = 1


class ScoutPlayer:

    def __init__(self, player_id):
        """
        Initialize a player.

        Args:
            player_id (int): The id of the player
        """
        self.player_id = player_id
        self.hand: list[ScoutCard] = []
        self.status = PlayerStatus.ALIVE

        # The chips that this player has put in until now
        self.scout_and_show = 0
        self.score = 0
        # self.money = 0

    def play_cards(self, start_idx, end_idx):
        new_set: list[ScoutCard] = self.hand[start_idx:end_idx]
        self.hand = self.hand[:start_idx] + self.hand[end_idx+1:]
        
        return new_set
    
    def insert_card(self, card: ScoutCard, idx):
        self.hand.insert(idx, card)

    def get_player_id(self):
        return self.player_id