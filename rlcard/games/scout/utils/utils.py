from __future__ import annotations

from typing import TypedDict

from ..card import ScoutCard as Card

class CardSegment(TypedDict):
    start: int
    end: int
    cards: list[Card]

ActionList = dict[str, int]

def init_deck() -> list[Card]:
    return [Card(top, bottom) for top in range(1, 11) for bottom in range(top + 1, 11)]

def is_valid_scout_segment(cards_in_segment: list[Card]) -> bool:
    """
    Returns True if the consecutive slice of the hand (cards_in_segment)
    forms a valid Scout set without reordering.
    A valid set is:
    - A group of 2 or more cards with the same rank.
    - A run of 2 or more cards with strictly ascending ranks (diff = +1).
    - A run of 2 or more cards with strictly descending ranks (diff = -1).
    """
    length = len(cards_in_segment)

    # Segments must generally have at least 2 cards to be a group or run.
    # Adjust this if single cards are considered valid segments in your rules.
    if length < 2:
        # If single cards were valid, you would return True here.
        # Based on the original logic checking pairs, assuming runs/groups need >= 2.
        return True 
    
    # 1. Check if they're all the same rank (group)
    first_rank = cards_in_segment[0].rank
    is_group = all(card.rank == first_rank for card in cards_in_segment)
    if is_group:
        return True

    # 2. If not a group, check for runs (ascending or descending)
    # We need to check if *all* differences are +1 OR *all* differences are -1.
    
    is_ascending_run = True
    is_descending_run = True

    for i in range(length - 1):
        curr_rank = cards_in_segment[i].rank
        next_rank = cards_in_segment[i+1].rank
        diff = next_rank - curr_rank

        # Check ascending condition
        if diff != 1:
            is_ascending_run = False
        
        # Check descending condition
        if diff != -1:
            is_descending_run = False

        # Optimization: if it's neither type of run so far, we can stop checking.
        if not is_ascending_run and not is_descending_run:
            return False 

    # If the loop completes, the segment is valid if it was *either*
    # a consistent ascending run OR a consistent descending run.
    # Note: A segment cannot be both simultaneously if length >= 2.
    return is_ascending_run or is_descending_run


def find_all_scout_segments(hand: list[Card]) -> list[CardSegment]:
    """
    Returns a list of all valid Scout segments (each segment is a contiguous
    slice of the hand). 
    Each element in the returned list is a dictionary containing:
        'start': starting index (inclusive)
        'end': ending index (exclusive)
        'cards': the list of ScoutCard objects in the segment
    """
    valid_segments: list[CardSegment] = []
    n = len(hand)

    # Explore all contiguous slices: i is start, j is end (exclusive)
    for i in range(n):
        # Check segments of length 2 or more
        for j in range(i + 2, n + 1):  # Start j from i+2 for length >= 2 slices
            # slice is hand[i:j], which includes i..(j-1)
            cards_in_segment = hand[i:j]
            
            # Check if this slice is a valid group or run
            if is_valid_scout_segment(cards_in_segment):
                valid_segments.append({
                    'start': i,
                    'end': j,
                    'cards': cards_in_segment
                })
                
    for i in range(n):
        valid_segments.append({'start': i, 'end': i+1, 'cards': [hand[i]]})
        
    return valid_segments


# Optional helper to extract the actual cards from these indices:
def get_segment_from_hand(hand: list[Card], start_idx: int, end_idx: int) -> list[Card]:
    return hand[start_idx:end_idx]

def compare_scout_segments(
    new_hand: list[Card],
    new_start: int,
    new_end: int,
    old_hand: list[Card],
    old_start: int,
    old_end: int,
) -> bool:
    """
    Compares two segments:
      - new_hand[new_start:new_end]
      - old_hand[old_start:old_end]
    Returns True if new segment is strictly stronger than old segment.
    """
    new_len = new_end - new_start
    old_len = old_end - old_start

    # 1) Compare lengths
    if new_len > old_len:
        return True
    elif new_len < old_len:
        return False

    # 2) Same length => compare highest rank
    new_segment = new_hand[new_start:new_end]
    old_segment = old_hand[old_start:old_end]

    # We'll define a helper to get the "strength rank" of a segment
    new_strength = segment_strength_rank(new_segment)
    old_strength = segment_strength_rank(old_segment)

    if new_strength > old_strength:
        return True
    else:
        return False


def segment_strength_rank(cards: list[Card]) -> tuple[int, int]:
    """
    Compute the 'strength rank' of a set according to Scout rules.
    Scout hierarchy (from strongest to weakest):
    1. Groups (matching cards) - highest rank wins
    2. Runs (consecutive cards) - highest rank wins
    3. Single cards - rank value
    
    Returns a tuple (type_priority, rank) where:
    - type_priority: 2 for groups, 1 for runs, 0 for single cards
    - rank: the highest rank in the set
    """
    length = len(cards)

    # Single card => its own rank
    if length == 1:
        return (0, cards[0].rank)

    # Check if it's a group (all same rank)
    if all(c.rank == cards[0].rank for c in cards):
        return (2, cards[0].rank)  # Group with priority 2

    # Check if it's a run (consecutive ranks)
    is_ascending = True
    is_descending = True
    
    for i in range(length - 1):
        diff = cards[i+1].rank - cards[i].rank
        if diff != 1:
            is_ascending = False
        if diff != -1:
            is_descending = False
        if not is_ascending and not is_descending:
            break
    
    if is_ascending or is_descending:
        # Run with priority 1, highest rank wins
        return (1, max(cards[0].rank, cards[-1].rank))
    
    # Fallback: not a valid set, but return something
    return (0, max(c.rank for c in cards))

def get_action_list(max_hand_size: int = 16) -> ActionList:
    ACTION_LIST: ActionList = {}
    # Fill ACTION_LIST with every possible (start_pos, end_pos) for playing
    for start_pos in range(max_hand_size):
        for end_pos in range(start_pos+1, max_hand_size+1):
            ACTION_LIST[f"play-{start_pos}-{end_pos}"] = len(ACTION_LIST)

    # Then fill in scout actions (both normal and flipped)
    for ins_idx in range(max_hand_size+1):
        ACTION_LIST[f"scout-front-{ins_idx}-normal"] = len(ACTION_LIST)
        ACTION_LIST[f"scout-front-{ins_idx}-flip"] = len(ACTION_LIST)
        ACTION_LIST[f"scout-back-{ins_idx}-normal"] = len(ACTION_LIST)
        ACTION_LIST[f"scout-back-{ins_idx}-flip"] = len(ACTION_LIST)

    return ACTION_LIST
