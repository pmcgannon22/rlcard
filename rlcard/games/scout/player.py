from __future__ import annotations

from enum import Enum

from rlcard.games.scout.card import ScoutCard

class PlayerStatus(Enum):
    ALIVE = 0
    OUT = 1


class ScoutPlayer:
    def __init__(self, player_id: int) -> None:
        """
        Initialize a player.

        Args:
            player_id (int): The id of the player
        """
        self.player_id: int = player_id
        self.hand: list[ScoutCard] = []
        self.status: PlayerStatus = PlayerStatus.ALIVE

        # The chips that this player has put in until now
        self.scout_and_show: int = 0
        self.score: int = 0
        # self.money = 0

    def play_cards(self, start_idx: int, end_idx: int) -> list[ScoutCard]:
        new_set: list[ScoutCard] = self.hand[start_idx:end_idx]
        self.hand = self.hand[:start_idx] + self.hand[end_idx:]
        
        return new_set
    
    def insert_card(self, card: ScoutCard, idx: int) -> None:
        self.hand.insert(idx, card)

    def get_player_id(self) -> int:
        return self.player_id
