from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from rlcard.games.scout.game import ScoutGame
from rlcard.games.scout.player import ScoutPlayer


class ScoutJudger:
    def __init__(self, game: ScoutGame) -> None:
        self.game = game

    @staticmethod
    def compute_rewards(players: Sequence[ScoutPlayer]) -> list[int]:
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        rewards: list[int] = []
        for player in players:
            tokens_earned = player.score
            hand_penalty = len(player.hand)
            rewards.append(tokens_earned - hand_penalty)

        return rewards

    def get_legal_actions(self) -> None:
        pass
