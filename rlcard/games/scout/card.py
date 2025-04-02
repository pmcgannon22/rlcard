class ScoutCard:
    def __init__(self, top, bottom):
        ''' Initialize the class of ScoutCard

        Args:
            top (int): The top number
            bottom (int): The bottom number
        '''
        self.top = top
        self.bottom = bottom
        self.rank = top
        self.str = self.get_str()

    def get_str(self):
        ''' Get the string representation of card

        Return:
            (str): The string of card's top and bottom
        '''
        return f"{self.top}/{self.bottom}"

    @staticmethod
    def print_cards(cards):
        ''' Print out card in a nice form

        Args:
            card (str or list): The string form or a list of a UNO card
        '''
        if isinstance(cards, str):
            cards = [cards]
        for i, card in enumerate(cards):
            print(card.get_str())

            if i < len(cards) - 1:
                print(', ', end='')