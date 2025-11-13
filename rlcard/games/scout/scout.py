from __future__ import annotations

import random
from typing import NamedTuple, Self

class Card:
    def __init__(self, top: int, bottom: int) -> None:
        self.top: int = top
        self.bottom: int = bottom

    def __str__(self) -> str:
        return f"({self.top}/{self.bottom})"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __sub__(self, other: Self) -> int:
        return self.top - other.top
    
    def __eq__(self, other: Self) -> bool:
        return self.top == other.top and self.bottom == other.bottom
    
    def is_match(self, other: Self) -> bool:
        return self.top == other.top
    
    def is_seq(self, other: Self) -> bool:
        return abs(other.top - self.top) == 1

class Deck:
    cards: list[Card]

    def __init__(self) -> None:
        self.cards = [Card(top, bottom) for top in range(1, 11) for bottom in range(top + 1, 11)]
        random.shuffle(self.cards)

    def draw(self) -> Card:
        return self.cards.pop()

    def __str__(self) -> str:
        return f"{len(self.cards)} cards: {','.join([str(card) for card in self.cards])}"
    
    def __repr__(self) -> str:
        return str(self)

def value_set(cards: list[Card]) -> int:
    if not cards:
        return 0
    
    score = len(cards) * 1000
    if cards[0].is_match(cards[-1]):
        score += 500
    score += sum([c.top for c in cards])

    return score

class BestSet(NamedTuple):
    start: int
    finish: int
    value: int
    cards: list[Card]

def find_best_set(hand: list[Card]) -> BestSet:
    # (start, finish, value)
    best_set = BestSet(start=0, finish=0, value=0, cards=[])
    sets: list[list[Card]] = []
    cur_set: list[Card] = []
    offset = 0
    
    for i, card in enumerate(hand):
        
        # print(f"{44} {card=}")

        if not cur_set:
            cur_set = [card]
            offset = 0
        elif len(cur_set) == 1:
            if cur_set.is_match(card):
                cur_set.append(card)
            elif cur_set[0].is_seq(card):
                cur_set.append(card)
                offset = card - cur_set[-1]
            else:
                cur_set = [card]
                offset = 0
        else:
            diff = card - cur_set[-1]
            if diff == offset:
                cur_set.append(card)
            elif abs(diff) == abs(offset):
                cur_set = [cur_set[-1], card]
                offset = -offset
            else:
                cur_set = [card]
                offset = 0
        
        value = value_set(cur_set)

        if value > best_set.value:
            best_set = BestSet(start=i-len(cur_set)+1, finish=i+1, value=value, cards=cur_set)
            # print(f"New best set: {best_set}")

    # print(f"The best set to play is: {best_set}")

    return best_set

def best_insert(scouted: Card, hand: list[Card]) -> tuple[int, Card, list[Card]]:
    value, place, card = 0, 0, scouted
    for i in range(len(hand)):
        best_set_top = find_best_set(hand[:i] + [scouted] + hand[i:])
        best_set_bottom = find_best_set(hand[:i] + [Card(scouted.bottom, scouted.top)] + hand[i:])
        if best_set_top.value > value:
            value, place, card = best_set_top.value, i, scouted
        elif best_set_bottom.value > value:
            value, place, card = best_set_bottom.value, i, Card(card.bottom, card.top)    
    
    return (value, card, hand[:place] + [card] + hand[place:])

class Player:
    name: str
    hand: list[Card]
    score: int

    def __init__(self, name: str):
        self.name = name
        self.hand = []
        self.score = 0

    def draw_hand(self, deck: Deck) -> None:
        self.hand = [deck.draw() for _ in range(12)]

    def get_best_set(self) -> BestSet:
        best_set = find_best_set(self.hand)

        return best_set

    def play_set(self, played_set: BestSet) -> None:
        self.hand = self.hand[:played_set.start] + self.hand[played_set.finish:]

    def scout_card(self, active_set: list[Card]) -> Card:
        if len(active_set) == 1:
            (_, card, self.hand) = best_insert(active_set[0], self.hand)
        else:
            (_, card, self.hand) = max(best_insert(active_set[0], self.hand), best_insert(active_set[-1], self.hand), key=lambda h: h[0])

        if card == active_set[0]:
            active_set.pop(0)
        else:
            active_set.pop(-1)
        
        return card
    
    def give_points(self, n: int) -> None:
        self.score += n

    def __repr__(self) -> str:
        return f"{self.name} hand: {' '.join(map(str, self.hand))}"

class ScoutGame:
    def __init__(self) -> None:
        self.deck = Deck()
        self.players: list[Player] = [Player("Patrick"), Player("Miriam")]
        self.current_set: list[Card] = []
        self.current_set_val: int = 0
        self.round_ended: bool = False

    def start_game(self) -> None:
        for player in self.players:
            player.draw_hand(self.deck)
        self.current_player_index: int = 0
        self.play_game()

    def switch_player(self) -> None:
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def play_game(self) -> None:
        while not self.round_ended:
            current_player = self.players[self.current_player_index]
            print(current_player)
            print(f"Current set on table: {self.current_set}")
            action = input(f"{current_player.name}, do you want to play a set or scout a card? (p/s): ")
            if action == "p":
                best_set = current_player.get_best_set()
                if best_set.value <= self.current_set_val:
                    print("\nYou can't play!\n")
                    continue

                current_player.play_set(best_set)
                current_player.give_points(len(self.current_set))

                self.current_set = best_set.cards
                self.current_set_val = best_set.value
            elif action == "s":
                scouted = current_player.scout_card(self.current_set)
                # current_player.score += 1
                print(f"{current_player.name} scouted: {scouted}")
            else:
                print("Invalid action. Try again.")

            if not current_player.hand:
                self.round_ended = True
            else:
                self.switch_player()

        self.calculate_scores()

    def calculate_scores(self) -> None:
        for player in self.players:
            print(f"{player.name}'s score: {player.score}")

# def find_sets_above_value(nums: List[int], V: int) -> List[Tuple[int]]:
#     def calculate_value(subset: List[int]) -> int:
#         length_bonus = len(subset) * 1000
#         sum_bonus = sum(subset)
#         same_bonus = 500 if all(x == subset[0] for x in subset) else 0
#         return length_bonus + sum_bonus + same_bonus

#     def add_valid_sets(start: int, end: int) -> None:
#         for k in range(start + 2, end + 1):
#             subset = nums[start:k]
#             value = calculate_value(subset)
#             if value > V:
#                 sets_above_value.append(tuple(subset))

#     sets_above_value = []
#     n = len(nums)

#     i = 0
#     while i < n:
#         # Check for all same
#         j = i
#         while j < n and nums[j] == nums[i]:
#             j += 1
#         add_valid_sets(i, j)
#         i = j
        
#         # Check for strictly increasing
#         if i < n:
#             j = i
#             while j + 1 < n and nums[j + 1] == nums[j] + 1:
#                 j += 1
#             add_valid_sets(i, j + 1)
#             i = j + 1
        
#         # Check for strictly decreasing
#         if i < n:
#             j = i
#             while j + 1 < n and nums[j + 1] == nums[j] - 1:
#                 j += 1
#             add_valid_sets(i, j + 1)
#             i = j + 1

#     return sets_above_value

def find_sets_above_value(hand: list[Card], min_val: int = 0) -> list[list[Card]]:
    # (start, finish, value)
    best_set = BestSet(start=0, finish=0, value=0, cards=[])
    sets = []
    cur_set: List[Card] = []
    
    i = 0
    while i < len(hand):
        n = hand[i]
        if len(cur_set):
            if n.top == cur_set[-1] and n.top == cur_set[0]:
                cur_set.append(n)
                i += 1
            elif n.top - cur_set[-1].top == 1:
                cur_set.append(n)
                i += 1
            else:
                cur_set = [cur_set[-1]]
                continue
        else:
            cur_set.append(n)
            i += 1

        if len(cur_set) > 1:
            sets.append(cur_set)
        if (v := value_set(cur_set)) > best_set.value:
            best_set = BestSet(start=i-len(cur_set)+1, finish=i, value=v, cards=cur_set)

    return sets


        # if not cur_set:
        #     cur_set = [card]
        #     offset = 0
        # elif len(cur_set) == 1:
        #     if cur_set[0].is_match(card):
        #         cur_set.append(card)
        #     elif cur_set[0].is_seq(card):
        #         cur_set.append(card)
        #         offset = card.top - cur_set[-1].top
        #     else:
        #         cur_set = [card]
        #         offset = 0
        # else:
        #     diff = card.top - cur_set[-1].top
        #     if diff == offset:
        #         cur_set.append(card)
        #     elif abs(diff) == abs(offset):
        #         cur_set = [cur_set[-1], card]
        #         offset = -offset
        #     else:
        #         cur_set = [card]
        #         offset = 0
        
    #     value = value_set(cur_set)

    #     if value > min_val:
    #         sets.append(cur_set)

    # return sets

if __name__ == "__main__":
    # deck = Deck()
    # print(deck)

    # pat = Player("Pat")
    # pat.draw_hand(deck)

    # print(f'{pat.hand=}')

    # while pat.hand:
    #     print(f'Current Hand: {pat.hand}\n')
    #     best = pat.get_best_set()
    #     print(f"best = {pat.hand[best[0]:best[1]]}")
    #     pat.play_set(best.start, best.finish)
    #     print("\n")
    #     input("Continue: ")
    #     print("\n")

    # game = ScoutGame()
    # game.start_game()

    nums = [3, 4, 5, 5, 6, 7, 7, 7, 8, 10, 9, 10, 8, 9]

    V = 1611
    print(find_sets_above_value([Card(n, 0) for n in nums], V))
