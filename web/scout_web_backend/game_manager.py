from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.dmc_agent.model import DMCModel
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
        }
        recent.append(entry)
    return recent


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
        agent.load_state_dict(state_dict)
        agent.set_device('cpu' if device == 'cpu' else f'cuda:{device}')
    model.eval()
    return model.get_agents()


@dataclass
class GameConfig:
    checkpoint: Optional[Path] = None
    human_position: int = 0
    device: str = 'cpu'


class ScoutWebGame:
    def __init__(self, config: GameConfig):
        self.config = config
        self.env = rlcard.make('scout')
        self.num_players = self.env.num_players
        self.human_position = max(0, min(config.human_position, self.num_players - 1))
        self.ai_agents = self._init_agents(config.checkpoint, config.device)
        self.human_advisor = (
            self.ai_agents[self.human_position] if config.checkpoint else None
        )
        self.state = None
        self.current_player = 0
        self.reset_game()

    def _init_agents(self, checkpoint: Optional[Path], device: str):
        if checkpoint:
            return _load_dmc_agents(self.env, checkpoint, device)
        return [RandomAgent(num_actions=self.env.num_actions) for _ in range(self.num_players)]

    def reset_game(self):
        self.state, self.current_player = self.env.reset()
        self._auto_play_until_human(self.state)

    def _auto_play_until_human(self, pending_state=None):
        state = pending_state
        while not self.env.is_over() and self.current_player != self.human_position:
            if state is None:
                state = self.env.get_state(self.current_player)
            agent = self.ai_agents[self.current_player]
            action = agent.step(state)
            state, self.current_player = self.env.step(action, agent.use_raw)
            state = None
        if not self.env.is_over():
            self.state = self.env.get_state(self.human_position)
        else:
            self.state = None

    def apply_human_action(self, action_id: int):
        if self.env.is_over():
            return self.serialize_state()
        legal_actions = self.state['raw_legal_actions'] if self.state else []
        legal_ids = {a.action_id for a in legal_actions}
        if action_id not in legal_ids:
            raise ValueError("Invalid action for current state")
        next_state, next_player = self.env.step(action_id)
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

        next_state, next_player = self.env.step(match, raw_action=True)
        self.current_player = next_player
        if self.env.is_over():
            self.state = None
        elif next_player == self.human_position:
            self.state = next_state
        else:
            self._auto_play_until_human(next_state)
        return self.serialize_state()

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

        if not game_over and self.current_player == self.human_position and self.state:
            legal_actions = self.state['raw_legal_actions']
            if self.human_advisor:
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
                    })
                    legal_payload.append({
                        "action_id": action.action_id,
                        "title": desc['title'],
                        "description": desc['description'],
                        "type": desc['type'],
                        "isSuggestion": is_suggestion,
                    })
                elif isinstance(action, ScoutAction):
                    scout_actions.append(action)
        suggested_id_payload = int(suggested_id) if suggested_id is not None else None
        if (
            suggested_id_payload is None
            and scout_actions
            and self.human_advisor is not None
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
        round_state = self.env.game.round
        for pid, player in enumerate(round_state.players):
            scores.append({
                "player": pid,
                "score": int(player.score),
                "hand_size": int(len(player.hand)),
                "payoff": payoffs[pid] if payoffs else None,
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

        if game_over:
            prompt = winner_text or "Round complete."
        elif self.current_player == self.human_position:
            prompt = "Your turn"
        else:
            prompt = "Waiting for opponents..."

        scout_info = self._build_scout_info(raw_state, scout_actions)

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
        }
        return payload

    def _build_scout_info(self, raw_state: Dict[str, Any], scout_actions: List[ScoutAction]):
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
            targets.append({
                "direction": direction,
                "card": _card_to_dict(card),
                "allowFlip": any(bool(a.flip) for a in relevant),
            })

        arrows = []
        for slot in insertion_slots:
            if slot == 0:
                label = "Before first card"
            elif slot >= len(raw_state.get('hand', [])):
                label = "After last card"
            else:
                label = f"Between {slot-1} & {slot}"
            arrows.append({"slot": slot, "label": label})

        action_entries = []
        for action in scout_actions:
            action_entries.append({
                "action_id": action.action_id,
                "direction": "front" if action.from_front else "back",
                "flip": bool(action.flip),
                "insertion": int(action.insertion_in_hand),
            })

        return {
            "canScout": True,
            "targets": targets,
            "insertionSlots": insertion_slots,
            "arrows": arrows,
            "actions": action_entries,
        }
