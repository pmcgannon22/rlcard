"""AI advisor functionality for the Scout web game."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from rlcard.agents.dmc_agent.model import DMCAgent


def get_orientation_advice(
    env,
    human_position: int,
    advisor_agent: Optional[DMCAgent],
) -> Optional[Dict[str, Any]]:
    """Get AI advisor's recommendation for hand orientation.

    Evaluates both keeping and reversing the hand orientation to determine
    which yields a higher expected value.

    Args:
        env: The RLCard environment.
        human_position: The player position index.
        advisor_agent: The AI agent to use for evaluation.

    Returns:
        Dictionary with recommendation ("keep", "reverse", or "either"),
        keep_value, and reverse_value. Returns None if advisor is unavailable
        or evaluation fails.
    """
    if advisor_agent is None:
        return None

    round_obj = env.game.round
    player = round_obj.players[human_position]
    original_hand = deepcopy(player.hand)
    locks = getattr(round_obj, "orientation_locked", None)
    original_lock = locks[human_position] if locks else False

    def evaluate(reverse: bool) -> Optional[float]:
        """Evaluate the expected value for a given orientation."""
        player.hand = deepcopy(original_hand)
        if locks is not None:
            locks[human_position] = True

        if reverse:
            player.hand = [card.flip() for card in player.hand]

        try:
            temp_state = env.get_state(human_position)
            _, info = advisor_agent.eval_step(temp_state)
            values = info.get('values', {})
            if not values:
                return None
            return float(max(values.values()))
        except Exception:
            return None
        finally:
            player.hand = deepcopy(original_hand)
            if locks is not None:
                locks[human_position] = original_lock

    keep_val = evaluate(False)
    reverse_val = evaluate(True)

    # Restore original state
    player.hand = deepcopy(original_hand)
    if locks is not None:
        locks[human_position] = original_lock

    if keep_val is None or reverse_val is None:
        return None

    diff = keep_val - reverse_val
    epsilon = 1e-3

    if abs(diff) <= epsilon:
        recommended = "either"
    else:
        recommended = "keep" if diff > 0 else "reverse"

    return {
        "recommendation": recommended,
        "keep_value": keep_val,
        "reverse_value": reverse_val,
    }


def get_suggested_action(advisor_agent: Optional[DMCAgent], state: Dict[str, Any]) -> Optional[int]:
    """Get the AI advisor's suggested action for the current state.

    Args:
        advisor_agent: The AI agent to use for suggestions.
        state: The current game state.

    Returns:
        The suggested action ID, or None if unavailable or error occurs.
    """
    if advisor_agent is None or state is None:
        return None

    try:
        suggested_id, _ = advisor_agent.eval_step(state)
        return suggested_id
    except Exception:
        return None
