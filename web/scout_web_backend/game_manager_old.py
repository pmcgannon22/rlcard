from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.dmc_agent.model import DMCModel, DMCAgent
from rlcard.games.scout.card import ScoutCard
from rlcard.games.scout.utils.action_event import PlayAction, ScoutAction, ScoutEvent


def _card_to_dict(card: ScoutCard, position: Optional[int] = None) -> Dict[str, Any]:
    return {
        "top": card.top,
        "bottom": card.bottom,
        "label": f"{card.top}/{card.bottom}",
        "position": position,
    }


def _describe_action(action: ScoutEvent, raw_state: Dict[str, Any]) -> Dict[str, str]:
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


def _recent_actions(action_recorder: List[Any]) -> List[Dict[str, Any]]:
    recent = []
    for entry in action_recorder[-6:]:
        if len(entry) == 3:
            player_id, action, context = entry
        else:
            player_id, action = entry
            context = {}
        label = action.get_action_repr() if hasattr(action, "get_action_repr") else str(action)
        entry = {
            "player": player_id,
            "label": label,
            "context": context or {},
            "value": context.get("value") if isinstance(context, dict) else None,
        }
        recent.append(entry)
    return recent


def _align_state_dict_inputs(agent: DMCAgent, state_dict: dict) -> dict:
    """Pad/crop the first FC layer if observation size changed."""
    weight_key = 'fc_layers.0.weight'
    bias_key = 'fc_layers.0.bias'
    if weight_key not in state_dict or bias_key not in state_dict:
        return state_dict
    target_weight = agent.net.fc_layers[0].weight
    target_bias = agent.net.fc_layers[0].bias
    weight = state_dict[weight_key]
    bias = state_dict[bias_key]
    tw_shape = target_weight.shape
    if weight.shape != tw_shape:
        # Align output dimension
        if weight.shape[0] != tw_shape[0]:
            if weight.shape[0] > tw_shape[0]:
                weight = weight[:tw_shape[0], :]
            else:
                pad = torch.zeros(tw_shape[0] - weight.shape[0], weight.shape[1], dtype=weight.dtype)
                weight = torch.cat([weight, pad], dim=0)
        # Align input dimension
        if weight.shape[1] != tw_shape[1]:
            if weight.shape[1] > tw_shape[1]:
                weight = weight[:, :tw_shape[1]]
            else:
                pad = torch.zeros(weight.shape[0], tw_shape[1] - weight.shape[1], dtype=weight.dtype)
                weight = torch.cat([weight, pad], dim=1)
    if bias.shape != target_bias.shape:
        if bias.shape[0] > target_bias.shape[0]:
            bias = bias[:target_bias.shape[0]]
        else:
            pad = torch.zeros(target_bias.shape[0] - bias.shape[0], dtype=bias.dtype)
            bias = torch.cat([bias, pad], dim=0)
    new_state = dict(state_dict)
    new_state[weight_key] = weight
    new_state[bias_key] = bias
    return new_state


def _load_dmc_agents(env, checkpoint_path: Path, device: str):
    data = torch.load(checkpoint_path, map_location='cpu')
    state_shape = env.state_shape
    action_shape = env.action_shape
    if action_shape[0] is None:
        action_shape = [[env.num_actions] for _ in range(env.num_players)]
    model = DMCModel(
        state_shape,
        action_shape,
        exp_epsilon=0.0,
        device=device,
    )
    state_dicts = data.get('model_state_dict')
    if not state_dicts:
        raise ValueError("Checkpoint is missing 'model_state_dict'")
    for pid, state_dict in enumerate(state_dicts):
        agent = model.get_agent(pid)
        aligned = _align_state_dict_inputs(agent, state_dict)
        agent.load_state_dict(aligned)
        agent.set_device('cpu' if device == 'cpu' else f'cuda:{device}')
    model.eval()
    return model.get_agents()


@dataclass
class GameConfig:
    checkpoint: Optional[Path] = None
    human_position: int = 0
    device: str = 'cpu'
    advisor_enabled: bool = True
    debug_enabled: bool = False


class ScoutWebGame:
    def __init__(self, config: GameConfig):
        self.config = config
        self.env = rlcard.make('scout')
        self.num_players = self.env.num_players
        self.human_position = max(0, min(config.human_position, self.num_players - 1))
        self.ai_agents = self._init_agents(config.checkpoint, config.device)
        self.advisor_enabled = config.advisor_enabled
        self.debug_enabled = config.debug_enabled
        self.human_advisor = (
            self.ai_agents[self.human_position]
            if config.checkpoint and self.advisor_enabled
            else None
        )
        self.state = None
        self.current_player = 0
        self.reset_game()

    def _init_agents(self, checkpoint: Optional[Path], device: str):
        if checkpoint:
            return _load_dmc_agents(self.env, checkpoint, device)
        return [RandomAgent(num_actions=self.env.num_actions) for _ in range(self.num_players)]

    def reset_game(self):
        first_state, self.current_player = self.env.reset()
        if self.human_position == self.current_player:
            self.state = first_state
        else:
            self.state = self.env.get_state(self.human_position)
        self._game_log: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_players": self.num_players,
            "actions": [],
        }
        if not self._human_orientation_pending():
            self._auto_play_until_human()

    def _auto_play_until_human(self, pending_state=None):
        state = pending_state
        while (
            not self.env.is_over()
            and self.current_player != self.human_position
            and not self._human_orientation_pending()
        ):
            actor_id = self.current_player
            if state is None:
                state = self.env.get_state(actor_id)
            action_values = self._action_values_for_state(actor_id, state)
            agent = self.ai_agents[self.current_player]
            action = agent.step(state)
            value = None
            if action_values:
                action_id = self._action_id_from(action)
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

    def apply_human_action(self, action_id: int):
        if self.env.is_over():
            return self.serialize_state()
        if self.state and self.state['raw_obs'].get('must_choose_orientation'):
            raise ValueError("Choose your hand orientation first.")
        legal_actions = self.state['raw_legal_actions'] if self.state else []
        legal_ids = {a.action_id for a in legal_actions}
        if action_id not in legal_ids:
            raise ValueError("Invalid action for current state")
        action_values = self._action_values_for_state(self.human_position, self.state)
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

    def apply_scout_choice(self, direction: str, insertion_index: int, flip: bool):
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

        action_values = self._action_values_for_state(self.human_position, self.state)
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

    def choose_orientation(self, reverse: bool):
        if self.env.is_over():
            return self.serialize_state()
        self.env.game.set_orientation(self.human_position, reverse)
        self.state = self.env.get_state(self.human_position)
        if not self._human_orientation_pending():
            self._auto_play_until_human()
        return self.serialize_state()

    def set_advisor_enabled(self, enabled: bool):
        self.advisor_enabled = enabled
        self.config.advisor_enabled = enabled
        if enabled and self.config.checkpoint:
            self.human_advisor = self.ai_agents[self.human_position]
        else:
            self.human_advisor = None

    def set_debug_enabled(self, enabled: bool):
        self.debug_enabled = enabled
        self.config.debug_enabled = enabled

    def serialize_state(self) -> Dict[str, Any]:
        game_over = self.env.is_over()
        state = self.state or self.env.get_state(self.human_position)
        raw_state = state['raw_obs']
        hand = [_card_to_dict(card, idx) for idx, card in enumerate(raw_state['hand'])]
        table = [_card_to_dict(card) for card in raw_state['table_set']]
        legal_payload = []
        suggested_id = None
        play_options = []
        scout_actions = []
        action_values = {}

        if not game_over and self.current_player == self.human_position and self.state:
            legal_actions = self.state['raw_legal_actions']
            action_values = self._action_values_for_state(self.human_position, self.state)
            if self.human_advisor and self.advisor_enabled:
                try:
                    suggested_id, _ = self.human_advisor.eval_step(self.state)
                except Exception:
                    suggested_id = None
            for action in legal_actions:
                if isinstance(action, PlayAction):
                    desc = _describe_action(action, raw_state)
                    is_suggestion = bool(suggested_id == action.action_id)
                    play_options.append({
                        "action_id": action.action_id,
                        "title": desc['title'],
                        "description": desc['description'],
                        "type": desc['type'],
                        "isSuggestion": is_suggestion,
                        "value": action_values.get(action.action_id),
                    })
                    legal_payload.append({
                        "action_id": action.action_id,
                        "title": desc['title'],
                        "description": desc['description'],
                        "type": desc['type'],
                        "isSuggestion": is_suggestion,
                        "value": action_values.get(action.action_id),
                    })
                elif isinstance(action, ScoutAction):
                    scout_actions.append(action)
        suggested_id_payload = int(suggested_id) if suggested_id is not None else None
        if (
            suggested_id_payload is None
            and scout_actions
            and self.human_advisor is not None
            and self.advisor_enabled
        ):
            try:
                suggested_id_payload, _ = self.human_advisor.eval_step(self.state)
                suggested_id_payload = int(suggested_id_payload)
            except Exception:
                suggested_id_payload = None

        payoffs = self.env.get_payoffs().tolist() if game_over else None
        if payoffs is not None:
            payoffs = [int(p) for p in payoffs]
        scores = []
        state_values = self._player_state_values() if self.debug_enabled else {}
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

        scout_info = self._build_scout_info(raw_state, scout_actions, action_values or None)
        orientation_advice = None
        if orientation_pending and self.advisor_enabled:
            orientation_advice = self._orientation_advice()

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
            "recent_actions": _recent_actions(self.env.action_recorder),
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

    def _human_orientation_pending(self) -> bool:
        round_obj = getattr(self.env.game, "round", None)
        if not round_obj or not hasattr(round_obj, "orientation_locked"):
            return False
        return not round_obj.orientation_locked[self.human_position]

    def _latest_action_context(self) -> Dict[str, Any]:
        if not self.env.action_recorder:
            return {}
        entry = self.env.action_recorder[-1]
        if len(entry) == 3 and isinstance(entry[2], dict):
            return dict(entry[2])
        return {}

    def _action_values_for_state(self, player_id: int, state: Optional[Dict[str, Any]]):
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

    def _action_id_from(self, action) -> Optional[int]:
        if isinstance(action, ScoutEvent):
            return int(action.action_id)
        try:
            return int(action)
        except Exception:
            return None

    def _annotate_last_action_value(self, value: Optional[float]):
        if value is None or not self.env.action_recorder:
            return
        entry = self.env.action_recorder[-1]
        if len(entry) < 3:
            return
        context = entry[2]
        if isinstance(context, dict):
            context['value'] = float(value)

    def _log_action(self, player_id, action, value: Optional[float] = None):
        if isinstance(action, ScoutEvent):
            label = action.get_action_repr()
        elif hasattr(action, 'get_action_repr'):
            label = action.get_action_repr()
        else:
            label = str(action)
        context = self._latest_action_context()
        log_entry = {
            "player": int(player_id),
            "label": label,
            "type": context.get("action_type"),
            "details": context or {},
        }
        if value is not None:
            log_entry["value"] = float(value)
        self._game_log["actions"].append(log_entry)

    def _write_game_log(self):
        if "payoffs" not in self._game_log:
            return
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "scout_games.ndjson"
        with log_file.open("a") as f:
            f.write(json.dumps(self._game_log) + "\n")

    def _orientation_advice(self):
        if self.human_advisor is None:
            return None
        round_obj = self.env.game.round
        player = round_obj.players[self.human_position]
        original_hand = deepcopy(player.hand)
        locks = getattr(round_obj, "orientation_locked", None)
        original_lock = locks[self.human_position] if locks else False

        def evaluate(reverse: bool):
            player.hand = deepcopy(original_hand)
            if locks is not None:
                locks[self.human_position] = True
            if reverse:
                player.hand = [card.flip() for card in player.hand]
            try:
                temp_state = self.env.get_state(self.human_position)
                _, info = self.human_advisor.eval_step(temp_state)
                values = info.get('values', {})
                if not values:
                    return None
                return float(max(values.values()))
            except Exception:
                return None
            finally:
                player.hand = deepcopy(original_hand)
                if locks is not None:
                    locks[self.human_position] = original_lock

        keep_val = evaluate(False)
        reverse_val = evaluate(True)
        player.hand = deepcopy(original_hand)
        if locks is not None:
            locks[self.human_position] = original_lock
        self.state = self.env.get_state(self.human_position)

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

    def _player_state_values(self):
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
            action_values = self._action_values_for_state(pid, state)
            if action_values:
                values[pid] = max(action_values.values())
        return values

    def _build_scout_info(
        self,
        raw_state: Dict[str, Any],
        scout_actions: List[ScoutAction],
        value_lookup: Optional[Dict[int, float]] = None,
    ):
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
                "card": _card_to_dict(card),
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
