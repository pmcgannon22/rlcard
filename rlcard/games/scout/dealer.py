from rlcard.games.scout.utils import init_deck

from rlcard.games.scout.card import ScoutCard

class ScoutDealer:
    ''' Initialize a scout dealer class
    '''
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_deck()
        self.shuffle()
        self.table = []

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, num: int) -> list[ScoutCard]:
        ''' Deal some cards from deck to one player

        Args:
            player (object): The object of ScoutPlayer
            num (int): The number of cards to be dealed
        '''
        # TODO: generator

        cards: list[ScoutCard] = []
        for _ in range(num):
            cards.append(self.deck.pop())
        return cards


## For test
#if __name__ == '__main__':
#    dealer = ScoutDealer()
#    for card in dealer.deck:
#        print(card.get_str())
#    print(len(dealer.deck))
