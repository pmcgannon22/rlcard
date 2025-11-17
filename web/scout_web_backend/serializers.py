"""Serialization utilities for Scout game state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rlcard.games.scout.card import ScoutCard
from rlcard.games.scout.utils.action_event import PlayAction, ScoutAction, ScoutEvent


def card_to_dict(card: ScoutCard, position: Optional[int] = None) -> Dict[str, Any]:
    """Convert a ScoutCard to a dictionary for JSON serialization.

    Args:
        card: The card to serialize.
        position: Optional position index in hand or table.

    Returns:
        Dictionary with card properties (top, bottom, label, position).
    """
    return {
        "top": card.top,
        "bottom": card.bottom,
        "label": f"{card.top}/{card.bottom}",
        "position": position,
    }


def describe_action(action: ScoutEvent, raw_state: Dict[str, Any]) -> Dict[str, str]:
    """Generate a human-readable description of an action.

    Args:
        action: The action event to describe.
        raw_state: The raw game state containing hand and table information.

    Returns:
        Dictionary with title, description, and type of the action.
    """
    description = action.get_action_repr()
    title = description

    if isinstance(action, PlayAction):
        cards = raw_state['hand'][action.start_idx:action.end_idx]
        labels = ", ".join(f"{c.top}/{c.bottom}" for c in cards)
        title = f"Play cards {action.start_idx}-{action.end_idx-1}"
        description = labels or description
        action_type = "play"
    elif isinstance(action, ScoutAction):
        direction = "front" if action.from_front else "back"
        flip = " (flipped)" if getattr(action, 'flip', False) else ""
        title = f"Scout from {direction}{flip}"
        if raw_state['table_set']:
            target = raw_state['table_set'][0] if action.from_front else raw_state['table_set'][-1]
            description = f"{target.top}/{target.bottom} → slot {action.insertion_in_hand}"
        else:
            description = f"Insert at {action.insertion_in_hand}"
        action_type = "scout"
    else:
        action_type = "other"

    return {"title": title, "description": description, "type": action_type}


def serialize_recent_actions(action_recorder: List[Any]) -> List[Dict[str, Any]]:
    """Serialize the most recent actions from the action recorder.

    Args:
        action_recorder: List of action entries from the game environment.

    Returns:
        List of dictionaries with player, label, context, and value information.
    """
    recent = []
    for entry in action_recorder[-6:]:
        if len(entry) == 3:
            player_id, action, context = entry
        else:
            player_id, action = entry
            context = {}

        label = action.get_action_repr() if hasattr(action, "get_action_repr") else str(action)
        entry_dict = {
            "player": player_id,
            "label": label,
            "context": context or {},
            "value": context.get("value") if isinstance(context, dict) else None,
        }
        recent.append(entry_dict)
    return recent
