"""Main Scout web game implementation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import rlcard
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.games.scout.utils.action_event import PlayAction, ScoutAction, ScoutEvent

from .advisor import get_orientation_advice, get_suggested_action
from .agent_loader import initialize_agents
from .config import GameConfig
from .scout_info import build_scout_info
from .serializers import card_to_dict, describe_action, serialize_recent_actions


class ScoutWebGame:
    """Manager for a Scout game with human and AI players.

    This class handles game state, AI agent management, action processing,
    and state serialization for the web UI.
    """

    def __init__(self, config: GameConfig) -> None:
        """Initialize the game with the given configuration.

        Args:
            config: Game configuration including checkpoint path, human position, etc.
        """
        self.config = config
        self.env = rlcard.make('scout')
        self.num_players = self.env.num_players
        self.human_position = max(0, min(config.human_position, self.num_players - 1))
        self.ai_agents = initialize_agents(self.env, config.checkpoint, config.device)
        self.advisor_enabled = config.advisor_enabled
        self.debug_enabled = config.debug_enabled

        self.human_advisor = (
            self.ai_agents[self.human_position]
            if config.checkpoint and self.advisor_enabled
            else None
        )

        self.state: Optional[Dict[str, Any]] = None
        self.current_player: int = 0
        self._game_log: Dict[str, Any] = {}
        self.reset_game()

    def reset_game(self) -> None:
        """Reset the game to initial state and start a new round."""
        first_state, self.current_player = self.env.reset()

        if self.human_position == self.current_player:
            self.state = first_state
        else:
            self.state = self.env.get_state(self.human_position)

        self._game_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_players": self.num_players,
            "actions": [],
        }

        if not self._is_human_orientation_pending():
            self._auto_play_until_human()

    def apply_human_action(self, action_id: int) -> Dict[str, Any]:
        """Apply a human player's action.

        Args:
            action_id: The action ID to execute.

        Returns:
            Serialized game state after the action.

        Raises:
            ValueError: If the action is invalid or orientation must be chosen first.
        """
        if self.env.is_over():
            return self.serialize_state()

        if self.state and self.state['raw_obs'].get('must_choose_orientation'):
            raise ValueError("Choose your hand orientation first.")

        legal_actions = self.state['raw_legal_actions'] if self.state else []
        legal_ids = {a.action_id for a in legal_actions}

        if action_id not in legal_ids:
            raise ValueError("Invalid action for current state")

        action_values = self._get_action_values(self.human_position, self.state)
        value = action_values.get(int(action_id)) if action_values else None

        next_state, next_player = self.env.step(action_id)
        self._log_action(self.current_player, action_id, value)
        self._annotate_last_action_value(value)
        self.current_player = next_player

        if self.env.is_over():
            self.state = None
        elif next_player == self.human_position:
            self.state = next_state
        else:
            self._auto_play_until_human(next_state)

        return self.serialize_state()

    def apply_scout_choice(
        self,
        direction: str,
        insertion_index: int,
        flip: bool
    ) -> Dict[str, Any]:
        """Apply a scout action with specific parameters.

        Args:
            direction: Either "front" or "back".
            insertion_index: Position in hand to insert the scouted card.
            flip: Whether to flip the card.

        Returns:
            Serialized game state after the action.

        Raises:
            ValueError: If the parameters don't match any legal scout action.
        """
        if self.env.is_over():
            return self.serialize_state()

        if self.current_player != self.human_position or self.state is None:
            raise ValueError("It is not the human's turn.")

        if self.state['raw_obs'].get('must_choose_orientation'):
            raise ValueError("Choose your hand orientation first.")

        legal_actions = [
            action for action in self.state['raw_legal_actions']
            if isinstance(action, ScoutAction)
        ]

        if not legal_actions:
            raise ValueError("Scouting is not allowed right now.")

        direction_map = {"front": True, "back": False}
        if direction not in direction_map:
            raise ValueError("Direction must be 'front' or 'back'.")

        match = None
        for action in legal_actions:
            if (
                action.from_front == direction_map[direction]
                and action.insertion_in_hand == insertion_index
                and bool(action.flip) == bool(flip)
            ):
                match = action
                break

        if match is None:
            raise ValueError("No legal scout action matches the provided selection.")

        action_values = self._get_action_values(self.human_position, self.state)
        action_id = match.action_id if isinstance(match, ScoutEvent) else None
        value = action_values.get(int(action_id)) if (action_values and action_id is not None) else None

        next_state, next_player = self.env.step(match, raw_action=True)
        self._log_action(self.current_player, match, value)
        self._annotate_last_action_value(value)
        self.current_player = next_player

        if self.env.is_over():
            self.state = None
        elif next_player == self.human_position:
            self.state = next_state
        else:
            self._auto_play_until_human(next_state)

        return self.serialize_state()

    def choose_orientation(self, reverse: bool) -> Dict[str, Any]:
        """Set the human player's hand orientation.

        Args:
            reverse: Whether to reverse (flip) the hand.

        Returns:
            Serialized game state after setting orientation.
        """
        if self.env.is_over():
            return self.serialize_state()

        self.env.game.set_orientation(self.human_position, reverse)
        self.state = self.env.get_state(self.human_position)

        if not self._is_human_orientation_pending():
            self._auto_play_until_human()

        return self.serialize_state()

    def set_advisor_enabled(self, enabled: bool) -> None:
        """Enable or disable the AI advisor.

        Args:
            enabled: Whether the advisor should be enabled.
        """
        self.advisor_enabled = enabled
        self.config.advisor_enabled = enabled

        if enabled and self.config.checkpoint:
            self.human_advisor = self.ai_agents[self.human_position]
        else:
            self.human_advisor = None

    def set_debug_enabled(self, enabled: bool) -> None:
        """Enable or disable debug mode (showing action values).

        Args:
            enabled: Whether debug mode should be enabled.
        """
        self.debug_enabled = enabled
        self.config.debug_enabled = enabled

    def serialize_state(self) -> Dict[str, Any]:
        """Serialize the current game state for the web UI.

        Returns:
            Dictionary containing all game state information for rendering.
        """
        game_over = self.env.is_over()
        state = self.state or self.env.get_state(self.human_position)
        raw_state = state['raw_obs']

        hand = [card_to_dict(card, idx) for idx, card in enumerate(raw_state['hand'])]
        table = [card_to_dict(card) for card in raw_state['table_set']]

        legal_payload = []
        suggested_id = None
        play_options = []
        scout_actions = []
        action_values = {}

        if not game_over and self.current_player == self.human_position and self.state:
            legal_actions = self.state['raw_legal_actions']
            action_values = self._get_action_values(self.human_position, self.state)

            if self.human_advisor and self.advisor_enabled:
                suggested_id = get_suggested_action(self.human_advisor, self.state)

            for action in legal_actions:
                if isinstance(action, PlayAction):
                    desc = describe_action(action, raw_state)
                    is_suggestion = bool(suggested_id == action.action_id)
                    play_option = {
                        "action_id": action.action_id,
                        "title": desc['title'],
                        "description": desc['description'],
                        "type": desc['type'],
                        "isSuggestion": is_suggestion,
                        "value": action_values.get(action.action_id),
                    }
                    play_options.append(play_option)
                    legal_payload.append(play_option)
                elif isinstance(action, ScoutAction):
                    scout_actions.append(action)

        suggested_id_payload = int(suggested_id) if suggested_id is not None else None

        if (
            suggested_id_payload is None
            and scout_actions
            and self.human_advisor is not None
            and self.advisor_enabled
        ):
            suggested_id_payload = get_suggested_action(self.human_advisor, self.state)
            if suggested_id_payload is not None:
                suggested_id_payload = int(suggested_id_payload)

        payoffs = self.env.get_payoffs().tolist() if game_over else None
        if payoffs is not None:
            payoffs = [int(p) for p in payoffs]

        scores = []
        state_values = self._get_all_player_state_values() if self.debug_enabled else {}
        round_state = self.env.game.round

        for pid, player in enumerate(round_state.players):
            scores.append({
                "player": pid,
                "score": int(player.score),
                "hand_size": int(len(player.hand)),
                "payoff": payoffs[pid] if payoffs else None,
                "state_value": state_values.get(pid),
            })

        num_cards = {
            str(pid): int(count)
            for pid, count in (raw_state.get('num_cards', {}) or {}).items()
        }

        winner_text = None
        if payoffs:
            max_payoff = max(payoffs)
            winners = [i for i, p in enumerate(payoffs) if p == max_payoff]
            if len(winners) == 1:
                winner_text = f"Player {winners[0]} wins with payoff {max_payoff}"
            else:
                winner_text = f"Players {', '.join(map(str, winners))} tie with payoff {max_payoff}"

        legal_available = bool(legal_payload)
        orientation_pending = bool(raw_state.get('must_choose_orientation'))

        if game_over:
            prompt = winner_text or "Round complete."
        elif self.current_player == self.human_position:
            prompt = "Your turn"
        else:
            prompt = "Waiting for opponents..."

        scout_info = build_scout_info(raw_state, scout_actions, action_values or None)
        orientation_advice = None

        if orientation_pending and self.advisor_enabled:
            orientation_advice = get_orientation_advice(
                self.env,
                self.human_position,
                self.human_advisor,
            )

        payload = {
            "game_over": bool(game_over),
            "human_position": self.human_position,
            "current_player": self.current_player,
            "hand": hand,
            "table": table,
            "legal_actions": legal_payload,
            "suggested_action_id": suggested_id_payload,
            "legal_actions_available": legal_available,
            "play_options": play_options,
            "scout_info": scout_info,
            "recent_actions": serialize_recent_actions(self.env.action_recorder),
            "scores": scores,
            "table_owner": (
                int(raw_state.get('table_owner')) if raw_state.get('table_owner') is not None else None
            ),
            "consecutive_scouts": int(raw_state.get('consecutive_scouts', 0)),
            "num_cards": num_cards,
            "payoffs": payoffs,
            "winner_text": winner_text,
            "action_prompt": prompt,
            "num_players": self.num_players,
            "orientation_pending": orientation_pending,
            "orientation_advice": orientation_advice if self.advisor_enabled else None,
            "advisor_enabled": self.advisor_enabled,
            "debug_enabled": self.debug_enabled,
        }

        if game_over:
            payload["actions"] = self._game_log["actions"]
            self._game_log["payoffs"] = payoffs
            self._game_log["winner_text"] = winner_text
            self._write_game_log()

        return payload

    def _auto_play_until_human(self, pending_state: Optional[Dict[str, Any]] = None) -> None:
        """Automatically play AI turns until it's the human's turn.

        Args:
            pending_state: Optional state to use for the first iteration.
        """
        state = pending_state

        while (
            not self.env.is_over()
            and self.current_player != self.human_position
            and not self._is_human_orientation_pending()
        ):
            actor_id = self.current_player

            if state is None:
                state = self.env.get_state(actor_id)

            action_values = self._get_action_values(actor_id, state)
            agent = self.ai_agents[self.current_player]
            action = agent.step(state)

            value = None
            if action_values:
                action_id = self._extract_action_id(action)
                if action_id is not None:
                    value = action_values.get(action_id)

            state, next_player = self.env.step(action, agent.use_raw)
            self._log_action(actor_id, action, value)
            self._annotate_last_action_value(value)
            self.current_player = next_player
            state = None

        if not self.env.is_over():
            self.state = self.env.get_state(self.human_position)
        else:
            self.state = None

    def _is_human_orientation_pending(self) -> bool:
        """Check if the human player needs to choose hand orientation.

        Returns:
            True if orientation choice is pending, False otherwise.
        """
        round_obj = getattr(self.env.game, "round", None)
        if not round_obj or not hasattr(round_obj, "orientation_locked"):
            return False
        return not round_obj.orientation_locked[self.human_position]

    def _get_action_values(
        self,
        player_id: int,
        state: Optional[Dict[str, Any]]
    ) -> Dict[int, float]:
        """Get Q-values for all actions in the current state.

        Args:
            player_id: The player whose actions to evaluate.
            state: The game state.

        Returns:
            Mapping from action_id to Q-value. Empty dict if unavailable.
        """
        if not self.debug_enabled or state is None:
            return {}

        if player_id < 0 or player_id >= len(self.ai_agents):
            return {}

        agent = self.ai_agents[player_id]
        if not hasattr(agent, "predict"):
            return {}

        try:
            action_keys, values = agent.predict(state)
        except Exception:
            return {}

        return {int(key): float(val) for key, val in zip(action_keys, values)}

    def _get_all_player_state_values(self) -> Dict[int, float]:
        """Get the maximum Q-value for each player's current state.

        Returns:
            Mapping from player_id to max Q-value.
        """
        if not self.debug_enabled:
            return {}

        values = {}
        for pid in range(self.num_players):
            if pid >= len(self.ai_agents):
                continue

            agent = self.ai_agents[pid]
            if not hasattr(agent, "predict"):
                continue

            try:
                state = self.env.get_state(pid)
            except Exception:
                continue

            action_values = self._get_action_values(pid, state)
            if action_values:
                values[pid] = max(action_values.values())

        return values

    def _extract_action_id(self, action: Any) -> Optional[int]:
        """Extract an integer action ID from various action types.

        Args:
            action: The action (can be ScoutEvent, int, etc.).

        Returns:
            Integer action ID, or None if extraction fails.
        """
        if isinstance(action, ScoutEvent):
            return int(action.action_id)
        try:
            return int(action)
        except Exception:
            return None

    def _get_latest_action_context(self) -> Dict[str, Any]:
        """Get the context dictionary from the most recent action.

        Returns:
            Context dictionary, or empty dict if unavailable.
        """
        if not self.env.action_recorder:
            return {}

        entry = self.env.action_recorder[-1]
        if len(entry) == 3 and isinstance(entry[2], dict):
            return dict(entry[2])

        return {}

    def _annotate_last_action_value(self, value: Optional[float]) -> None:
        """Add the Q-value to the most recent action in the recorder.

        Args:
            value: The Q-value to add.
        """
        if value is None or not self.env.action_recorder:
            return

        entry = self.env.action_recorder[-1]
        if len(entry) < 3:
            return

        context = entry[2]
        if isinstance(context, dict):
            context['value'] = float(value)

    def _log_action(
        self,
        player_id: int,
        action: Any,
        value: Optional[float] = None
    ) -> None:
        """Log an action to the game log.

        Args:
            player_id: The player who took the action.
            action: The action taken.
            value: Optional Q-value for the action.
        """
        if isinstance(action, ScoutEvent):
            label = action.get_action_repr()
        elif hasattr(action, 'get_action_repr'):
            label = action.get_action_repr()
        else:
            label = str(action)

        context = self._get_latest_action_context()
        log_entry = {
            "player": int(player_id),
            "label": label,
            "type": context.get("action_type"),
            "details": context or {},
        }

        if value is not None:
            log_entry["value"] = float(value)

        self._game_log["actions"].append(log_entry)

    def _write_game_log(self) -> None:
        """Write the completed game log to a JSON lines file."""
        if "payoffs" not in self._game_log:
            return

        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "scout_games.ndjson"

        with log_file.open("a") as f:
            f.write(json.dumps(self._game_log) + "\n")
