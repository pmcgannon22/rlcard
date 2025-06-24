from typing import Self

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

    def flip(self):
        ''' Flip the card, swapping top and bottom values
        
        Returns:
            (ScoutCard): A new card with top and bottom swapped
        '''
        return ScoutCard(self.bottom, self.top)

    def get_str(self):
        ''' Get the string representation of card

        Return:
            (str): The string of card's top and bottom
        '''
        return f"{self.top}/{self.bottom}"

    @staticmethod
    def print_cards(cards: list[Self]):
        ''' Print out card(s) in a nice form

        Args:
            cards (ScoutCard or list): A single card or a list of ScoutCards
        '''
        if not isinstance(cards, list):
            cards = [cards]
        
        output = []
        for card in cards:
            if isinstance(card, ScoutCard):
                output.append(card.get_str())
            else:
                # Handle potential errors if non-ScoutCard objects are passed
                output.append(str(card)) 
                
        print(', '.join(output))

    def __repr__(self):
        return self.get_str()
    
    def __str__(self):
        return self.get_str()