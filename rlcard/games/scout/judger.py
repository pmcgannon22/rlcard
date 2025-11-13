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
        largest_hand = max([len(player.hand) for player in players])
        
        # Calculate final rewards: points earned during gameplay + hand size bonus
        rewards: list[int] = []
        for player in players:
            # Points from gameplay (beating sets, forcing scouts)
            gameplay_points = player.score
            
            # Bonus for having fewer cards (closer to winning)
            hand_bonus = largest_hand - len(player.hand)
            
            # Total reward
            total_reward = gameplay_points + hand_bonus
            rewards.append(total_reward)
            
        return rewards

    def get_legal_actions(self) -> None:
        pass
