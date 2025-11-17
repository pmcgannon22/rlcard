"""Scout action information builder for the web UI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rlcard.games.scout.utils.action_event import ScoutAction

from .serializers import card_to_dict


def build_scout_info(
    raw_state: Dict[str, Any],
    scout_actions: List[ScoutAction],
    value_lookup: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Build comprehensive scout action information for the UI.

    Args:
        raw_state: The raw game state containing hand and table information.
        scout_actions: List of legal scout actions.
        value_lookup: Optional mapping from action_id to Q-value.

    Returns:
        Dictionary containing scout targets, insertion slots, arrows, and actions.
    """
    if not scout_actions:
        return {
            "canScout": False,
            "targets": [],
            "insertionSlots": [],
            "arrows": [],
            "actions": [],
        }

    insertion_slots = sorted({int(action.insertion_in_hand) for action in scout_actions})
    if not insertion_slots:
        insertion_slots = [0]

    hand_empty = len(raw_state.get('hand', [])) == 0
    if hand_empty:
        insertion_slots = [0]

    targets = []
    table_set = raw_state.get('table_set', [])

    for direction, from_front in (("front", True), ("back", False)):
        relevant = [a for a in scout_actions if a.from_front == from_front]
        if not relevant:
            continue

        card = table_set[0] if from_front else table_set[-1]
        direction_value = None

        if value_lookup:
            direction_value = max(
                (value_lookup.get(a.action_id) for a in relevant if value_lookup.get(a.action_id) is not None),
                default=None,
            )

        targets.append({
            "direction": direction,
            "card": card_to_dict(card),
            "allowFlip": any(bool(a.flip) for a in relevant),
            "value": direction_value,
        })

    arrows = []
    for slot in insertion_slots:
        if slot == 0:
            label = "Before first card"
        elif slot >= len(raw_state.get('hand', [])):
            label = "After last card"
        else:
            label = f"Between {slot-1} & {slot}"

        slot_value = None
        if value_lookup:
            slot_value = max(
                (
                    value_lookup.get(a.action_id)
                    for a in scout_actions
                    if a.insertion_in_hand == slot
                    and value_lookup.get(a.action_id) is not None
                ),
                default=None,
            )

        arrows.append({"slot": slot, "label": label, "value": slot_value})

    action_entries = []
    for action in scout_actions:
        action_entries.append({
            "action_id": action.action_id,
            "direction": "front" if action.from_front else "back",
            "flip": bool(action.flip),
            "insertion": int(action.insertion_in_hand),
            "value": value_lookup.get(action.action_id) if value_lookup else None,
        })

    return {
        "canScout": True,
        "targets": targets,
        "insertionSlots": insertion_slots,
        "arrows": arrows,
        "actions": action_entries,
    }
