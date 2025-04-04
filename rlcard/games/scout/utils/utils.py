from ..card import ScoutCard as Card

def init_deck() -> list[Card]:
    return [Card(top, bottom) for top in range(1, 11) for bottom in range(top + 1, 11)]

def is_valid_scout_segment(cards_in_segment):
    """
    Returns True if the consecutive slice of the hand (cards_in_segment)
    forms a valid Scout set: either a run or a group, *without reordering*.
    """
    # For Scout, usually sets must be at least length 2 (or 3, depending on your variant).
    # """
    # A Scout set can be (depending on variant):
    # - A single card (often allowed in Scout).
    # - A run of length >= 2 (strictly ascending ranks, in consecutive hand positions).
    # - A group of length >= 2 (identical ranks, in consecutive hand positions).
    # Returns True if `chosen_cards` form a valid set.
    # """
    # length = len(chosen_cards)
    # if length == 0:
    #     return False

    # # 1) Single card allowed?
    # if length == 1:
    #     return True

    # # 2) Check if all same rank (group)
    # #    For example, [7,7,7]
    # same_rank = all(c.rank == chosen_cards[0].rank for c in chosen_cards)
    # if same_rank:
    #     return True

    # # 3) Check for ascending run in the order given (no re-sorting!)
    # #    e.g., ranks [3,4,5] => differences are [1,1]
    # for idx in range(length - 1):
    #     if chosen_cards[idx+1].rank - chosen_cards[idx].rank != 1:
    #         return False

    # return True
    # if len(cards_in_segment) < 2:
        # return False
    
    # Check if they're all the same rank (group)
    if all(card.rank == cards_in_segment[0].rank for card in cards_in_segment):
        return True

    # Check for ascending run: strictly increasing by 1 rank each step,
    # in the order they appear (no sorting).
    for i in range(len(cards_in_segment) - 1):
        curr_rank = cards_in_segment[i].rank
        next_rank = cards_in_segment[i+1].rank
        if next_rank - curr_rank != 1:
            return False

    return True


def find_all_scout_segments(hand: list[Card]) -> tuple[int, int, list[Card]]:
    """
    Returns a list of all valid Scout segments (each segment is a contiguous
    slice of the hand). 
    Each element in the returned list will be a tuple: (start_index, end_index).
    The slice is [start_index, start_index+1, ..., end_index-1].
    """
    valid_segments = []
    n = len(hand)

    # Explore all contiguous slices: i is start, j is end (exclusive)
    for i in range(n):
        for j in range(i+1, n+1):  
            # slice is hand[i:j], which includes i..(j-1)
            cards_in_segment = hand[i:j]
            if is_valid_scout_segment(cards_in_segment):
                # We'll store (start, end) or just the list of cards
                valid_segments.append((i, j, cards_in_segment))
    return valid_segments


# Optional helper to extract the actual cards from these indices:
def get_segment_from_hand(hand, start_idx, end_idx):
    return hand[start_idx:end_idx]

def compare_scout_segments(
    new_hand, new_start, new_end, 
    old_hand, old_start, old_end
):
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


def segment_strength_rank(cards: list[Card]):
        """
        Compute the 'strength rank' of a set. For runs, we assume the last card
        is the highest rank (since it's ascending). For a group, all are the same,
        so just return that rank. For a single card, return its rank.
        """
        length = len(cards)

        # Single card => its own rank
        if length == 1:
            return cards[0].rank

        # Group => all same, so pick the first
        if all(c.rank == cards[0].rank for c in cards):
            return cards[0].rank

        # Run => ranks ascending in the order they appear
        # The last card has the highest rank
        return max(cards[0].rank, cards[-1].rank)

def get_action_list():
    MAX_HAND = 16
    ACTION_LIST = {}
    # Fill ACTION_LIST with every possible (start_pos, end_pos) for playing
    for start_pos in range(MAX_HAND):
        for end_pos in range(start_pos+1, MAX_HAND+1):
            ACTION_LIST[f"play-{start_pos}-{end_pos}"] = len(ACTION_LIST)

    # Then fill in scout actions
    for ins_idx in range(MAX_HAND+1):
        ACTION_LIST[f"scout-front-{ins_idx}"] = len(ACTION_LIST)
        ACTION_LIST[f"scout-back-{ins_idx}"] = len(ACTION_LIST)

    return ACTION_LIST