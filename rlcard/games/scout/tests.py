from __future__ import annotations

# Adjust these import lines to match your project's file names and structure:
# For example, if your utilities are in 'utils.py' and the card class is in 'scout.py'
from .utils import ActionList, find_all_scout_segments, compare_scout_segments, get_action_list
from .card import ScoutCard  # or wherever ScoutCard is defined

def test_find_all_segments(label: str, hand: list[ScoutCard]) -> None:
    """Helper to print out all valid segments found in a hand."""
    print(f"\n=== {label} ===")
    print("Hand (in order):", [c.get_str() for c in hand])
    segments = find_all_scout_segments(hand)
    print("Found valid contiguous segments (by indices):", segments)
    for (start_idx, end_idx) in segments:
        segment_cards = hand[start_idx:end_idx]
        segment_strs = [card.get_str() for card in segment_cards]
        print(f"  Segment {start_idx}:{end_idx} => {segment_strs}")


def test_compare_segments(
    label: str,
    hand: list[ScoutCard],
    new_start: int,
    new_end: int,
    old_start: int,
    old_end: int,
) -> None:
    """Helper to compare two segments (new vs old) from the same hand."""
    print(f"\n=== {label} ===")
    new_segment_cards = hand[new_start:new_end]
    old_segment_cards = hand[old_start:old_end]

    new_segment_strs = [c.get_str() for c in new_segment_cards]
    old_segment_strs = [c.get_str() for c in old_segment_cards]

    print(f"New segment [{new_start}:{new_end}]: {new_segment_strs}")
    print(f"Old segment [{old_start}:{old_end}]: {old_segment_strs}")

    is_stronger = compare_scout_segments(hand, new_start, new_end,
                                         hand, old_start, old_end)
    print("Result: Is new segment stronger than old segment? =>", is_stronger)


def run_tests() -> None:
    """
    Run a series of test scenarios on find_all_scout_segments and compare_scout_segments.
    """

    # 1) Basic run & group test
    hand1: list[ScoutCard] = [
        ScoutCard(3, 1),   # rank=3
        ScoutCard(4, 5),   # rank=4
        ScoutCard(5, 9),   # rank=5
        ScoutCard(5, 2),   # rank=5
        ScoutCard(6, 7)    # rank=6
    ]
    test_find_all_segments("Test 1: Basic Mixed Hand", hand1)

    # Compare [3,4,5] (indices 0..3) vs [5,5] (indices 2..4)
    # i.e. "Is [3,4,5] stronger than [5,5]?"
    test_compare_segments("Test 1 Compare Run vs Group", hand1, 0, 3, 2, 4)

    # 2) Hand with multiple groups
    hand2: list[ScoutCard] = [
        ScoutCard(7, 3),
        ScoutCard(7, 7),
        ScoutCard(7, 4),
        ScoutCard(8, 1),
        ScoutCard(8, 8),
        ScoutCard(9, 6),
    ]
    test_find_all_segments("Test 2: Multiple Groups & Partial Runs", hand2)

    # Compare a 3-of-a-kind [7,7,7] vs a potential run [7,8,9]
    # Indices for 3-of-a-kind: [0..3] => (0,1,2)
    # Indices for run: [2..5] => (2,3,4) is not a real run if they skip a card, so let's see what's valid
    # We'll just pretend we want to compare 0..3 vs. 2..5 for demonstration
    test_compare_segments("Test 2 Compare Group vs Run", hand2, 0, 3, 2, 5)

    # 3) Single-card tests (if single cards are allowed in your variant)
    hand3: list[ScoutCard] = [
        ScoutCard(10, 2),
        ScoutCard(11, 11),
        ScoutCard(12, 6)
    ]
    test_find_all_segments("Test 3: Single Card Allowed?", hand3)
    # Compare single card [10] with single card [11]
    test_compare_segments("Test 3 Compare Single Cards", hand3, 0, 1, 1, 2)

    # 4) Edge case: empty hand (no segments)
    hand4: list[ScoutCard] = []
    test_find_all_segments("Test 4: Empty Hand", hand4)
    # There's nothing to compare, so skip compare test

    # 5) Large hand with consecutive run
    hand5: list[ScoutCard] = [
        ScoutCard(1, 13),
        ScoutCard(2, 12),
        ScoutCard(3, 10),
        ScoutCard(4, 6),
        ScoutCard(5, 7),
        ScoutCard(6, 6),
        ScoutCard(7, 1),
        ScoutCard(8, 9)
    ]
    test_find_all_segments("Test 5: Long Consecutive Run", hand5)
    # Compare a big run [1..8] vs a smaller run [3..5]
    test_compare_segments("Test 5 Compare Long vs Short Run", hand5, 0, 5, 2, 5)

    # 6) Basic run & group test
    hand6: list[ScoutCard] = [
        ScoutCard(3, 1),   # rank=3
        ScoutCard(4, 5),   # rank=4
        ScoutCard(5, 9),   # rank=5
        ScoutCard(5, 2),   # rank=5
        ScoutCard(4, 7)    # rank=4
    ]
    test_find_all_segments("Test 6: Basic Mixed Hand Up & Down", hand6)


if __name__ == "__main__":
    # run_tests()
    actions: ActionList = get_action_list(max_hand_size=16)
    for action in actions:
        print(action)
