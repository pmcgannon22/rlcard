#!/usr/bin/env python3
"""
Complete Agent Game Logger - Records Scout games in LLM-friendly format with action decoding
"""

import json
import datetime
import argparse
import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.games.scout.card import ScoutCard
from rlcard.games.scout.utils.action_event import ScoutEvent

class GameLogger:
    def __init__(self, log_file="agent_games.log"):
        self.log_file = log_file
        
    def log_game(self, game_data):
        """Log a complete game"""
        timestamp = datetime.datetime.now().isoformat()
        game_log = {
            "timestamp": timestamp,
            "game_id": game_data["game_id"],
            "agents": game_data["agents"],
            "moves": game_data["moves"],
            "final_state": game_data["final_state"],
            "payoffs": game_data["payoffs"],
            "winner": game_data["winner"],
            "game_summary": game_data["game_summary"]
        }
        
        # Write to file immediately
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(game_log, indent=2, ensure_ascii=False, default=str) + '\n\n')

def card_to_str(card):
    """Convert ScoutCard to string representation"""
    if isinstance(card, ScoutCard):
        return f"{card.top}/{card.bottom}"
    return str(card)

def decode_action(action_id, state=None):
    """Decode action ID to human-readable format"""
    try:
        action = ScoutEvent.from_action_id(action_id)
        
        if hasattr(action, 'start_idx') and hasattr(action, 'end_idx'):
            # Play action
            if state and 'hand' in state and len(state['hand']) > 0:
                start_idx = min(action.start_idx, len(state['hand'])-1)
                end_idx = min(action.end_idx, len(state['hand']))
                if start_idx < end_idx:
                    cards = state['hand'][start_idx:end_idx]
                    card_strs = [card_to_str(card) for card in cards]
                    return {
                        "type": "play",
                        "description": f"Play cards at positions {action.start_idx}-{action.end_idx-1}: {', '.join(card_strs)}",
                        "start_idx": action.start_idx,
                        "end_idx": action.end_idx,
                        "cards": card_strs
                    }
            return {
                "type": "play",
                "description": f"Play cards at positions {action.start_idx}-{action.end_idx-1}",
                "start_idx": action.start_idx,
                "end_idx": action.end_idx
            }
        
        elif hasattr(action, 'from_front') and hasattr(action, 'insertion_in_hand'):
            # Scout action
            direction = "front" if action.from_front else "back"
            flip_str = " (flipped)" if getattr(action, 'flip', False) else ""
            
            if state and 'table_set' in state and state['table_set']:
                if action.from_front:
                    scout_card = state['table_set'][0]
                else:
                    scout_card = state['table_set'][-1]
                card_str = card_to_str(scout_card)
                return {
                    "type": "scout",
                    "description": f"Scout {card_str}{flip_str} from {direction}, insert at position {action.insertion_in_hand}",
                    "from_front": action.from_front,
                    "insertion_in_hand": action.insertion_in_hand,
                    "flip": getattr(action, 'flip', False),
                    "card_scouted": card_str
                }
            else:
                return {
                    "type": "scout",
                    "description": f"Scout from {direction}, insert at position {action.insertion_in_hand}{flip_str}",
                    "from_front": action.from_front,
                    "insertion_in_hand": action.insertion_in_hand,
                    "flip": getattr(action, 'flip', False)
                }
        
        return {
            "type": "unknown",
            "description": str(action),
            "action_id": action_id
        }
    except Exception as e:
        return {
            "type": "error",
            "description": f"Could not decode action {action_id}: {str(e)}",
            "action_id": action_id
        }

def log_agent_games(num_games=10, agents_config="dqn_vs_random", log_file="agent_games.log"):
    """Log multiple games between agents"""
    
    logger = GameLogger(log_file)
    
    # Load DQN model
    device = 'cpu'
    dqn_agent = torch.load('experiments/scout_dqn_restart/model.pth', weights_only=False, map_location=device)
    dqn_agent.set_device(device)
    
    # Create environment
    env = rlcard.make('scout')
    
    # Set up agents based on configuration
    if agents_config == "dqn_vs_random":
        agents = [dqn_agent, RandomAgent(env.num_actions), RandomAgent(env.num_actions), RandomAgent(env.num_actions)]
        agent_names = ["DQN", "Random", "Random", "Random"]
    elif agents_config == "dqn_vs_dqn":
        agents = [dqn_agent, dqn_agent, dqn_agent, dqn_agent]
        agent_names = ["DQN", "DQN", "DQN", "DQN"]
    elif agents_config == "random_vs_random":
        agents = [RandomAgent(env.num_actions) for _ in range(4)]
        agent_names = ["Random", "Random", "Random", "Random"]
    else:
        raise ValueError(f"Unknown agents_config: {agents_config}")
    
    env.set_agents(agents)
    
    print(f"Logging {num_games} games with configuration: {agents_config}")
    print(f"Log file: {log_file}")
    print("=" * 60)
    
    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}")
        
        # Initialize game
        state, player_id = env.reset()
        game_data = {
            "game_id": game_num + 1,
            "agents": agent_names,
            "moves": [],
            "final_state": None,
            "payoffs": None,
            "winner": None,
            "game_summary": None
        }
        
        # Play the game
        while not env.is_over():
            current_player = player_id
            current_state = env.get_state(current_player)
            
            # Get action from current player
            if hasattr(agents[current_player], 'eval_step'):
                action, _ = agents[current_player].eval_step(current_state)
            else:
                action = agents[current_player].step(current_state)
            
            # Decode action
            decoded_action = decode_action(action, current_state)
            
            # Log the move
            move_data = {
                "player_id": int(current_player),
                "agent_name": agent_names[current_player],
                "action_id": int(action),
                "action": decoded_action,
                "state_before": {
                    "hand_size": len(current_state.get("hand", [])),
                    "table_set_size": len(current_state.get("table_set", [])),
                    "table_owner": current_state.get("table_owner"),
                    "consecutive_scouts": current_state.get("consecutive_scouts", 0),
                    "score": current_state.get("points", 0)
                }
            }
            game_data["moves"].append(move_data)
            
            # Take the action
            next_state, next_player = env.step(action)
            player_id = next_player
        
        # Log final state and payoffs
        game_data["final_state"] = {
            "hand_sizes": [len(env.game.round.players[i].hand) for i in range(4)],
            "scores": [env.game.round.players[i].score for i in range(4)],
            "table_set": [card_to_str(card) for card in env.game.round.table_set] if env.game.round.table_set else []
        }
        game_data["payoffs"] = [float(p) for p in env.get_payoffs()]
        game_data["winner"] = int(max(range(4), key=lambda i: game_data["payoffs"][i]))
        
        # Create game summary
        total_moves = len(game_data["moves"])
        play_moves = sum(1 for move in game_data["moves"] if move["action"]["type"] == "play")
        scout_moves = sum(1 for move in game_data["moves"] if move["action"]["type"] == "scout")
        
        game_data["game_summary"] = {
            "total_moves": total_moves,
            "play_moves": play_moves,
            "scout_moves": scout_moves,
            "winner_payoff": game_data["payoffs"][game_data["winner"]],
            "final_hand_sizes": game_data["final_state"]["hand_sizes"],
            "final_scores": game_data["final_state"]["scores"]
        }
        
        # Log the complete game
        logger.log_game(game_data)
        
        # Print summary
        winner = game_data["winner"]
        print(f"  Winner: Player {winner} ({agent_names[winner]}) - Payoff: {game_data['payoffs'][winner]:.2f}")
        print(f"  Moves: {total_moves} total ({play_moves} plays, {scout_moves} scouts)")
    
    print("=" * 60)
    print(f"Logged {num_games} games to {log_file}")
    
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Scout agent games for LLM analysis")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games to log")
    parser.add_argument("--agents_config", type=str, default="dqn_vs_random", 
                       choices=["dqn_vs_random", "dqn_vs_dqn", "random_vs_random"],
                       help="Agent configuration")
    parser.add_argument("--log_file", type=str, default="agent_games.log", help="Log file path")
    
    args = parser.parse_args()
    
    logger = log_agent_games(args.num_games, args.agents_config, args.log_file)
