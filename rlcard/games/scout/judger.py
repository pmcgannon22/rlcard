
class ScoutJudger:
    def __init__(self, game):
        self.game = game

    @staticmethod
    def compute_rewards(players):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        largest_hand = max([len(player.hand) for player in players])

        return [largest_hand - len(player.hand) for player in players]

    def get_legal_actions(self):
        pass