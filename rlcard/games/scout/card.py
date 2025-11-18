from __future__ import annotations

try:
    from typing import Self  # Python 3.11+
except ImportError:  # pragma: no cover
    from typing_extensions import Self


class ScoutCard:
    def __init__(self, top: int, bottom: int) -> None:
        ''' Initialize the class of ScoutCard

        Args:
            top (int): The top number
            bottom (int): The bottom number
        '''
        self.top: int = top
        self.bottom: int = bottom
        self.rank: int = top
        self.str: str = self.get_str()

    def flip(self) -> Self:
        ''' Flip the card, swapping top and bottom values
        
        Returns:
            (ScoutCard): A new card with top and bottom swapped
        '''
        return ScoutCard(self.bottom, self.top)

    def get_str(self) -> str:
        ''' Get the string representation of card

        Return:
            (str): The string of card's top and bottom
        '''
        return f"{self.top}/{self.bottom}"

    @staticmethod
    def print_cards(cards: ScoutCard | list[ScoutCard]) -> None:
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

    def __repr__(self) -> str:
        return self.get_str()
    
    def __str__(self) -> str:
        return self.get_str()
