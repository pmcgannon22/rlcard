# Scout Agent Game Logging System

This directory contains scripts for logging Scout agent gameplay in a format optimized for LLM analysis.

## Files

- `log_agent_games.py` - Main logging script with detailed action decoding
- `log_games.sh` - Shell script wrapper for easy execution

## Usage

### Quick Start
```bash
# Log 10 games of DQN vs Random agents
./examples/scripts/log_games.sh

# Log 50 games of DQN vs Random agents
./examples/scripts/log_games.sh 50

# Log 100 games of DQN self-play
./examples/scripts/log_games.sh 100 dqn_vs_dqn

# Custom log file
./examples/scripts/log_games.sh 25 dqn_vs_random logs/my_games.log
```

### Direct Python Usage
```bash
python3 examples/scripts/log_agent_games.py \
    --num_games 50 \
    --agents_config dqn_vs_random \
    --log_file logs/agent_games.log
```

## Agent Configurations

- `dqn_vs_random` - DQN agent vs 3 Random agents
- `dqn_vs_dqn` - 4 DQN agents (self-play)
- `random_vs_random` - 4 Random agents

## Log Format

Each game is logged as a JSON object with the following structure:

```json
{
  "timestamp": "2025-06-23T16:55:09.351663",
  "game_id": 1,
  "agents": ["DQN", "Random", "Random", "Random"],
  "moves": [
    {
      "player_id": 0,
      "agent_name": "DQN",
      "action_id": 82,
      "action": {
        "type": "play",
        "description": "Play cards at positions 6-7",
        "start_idx": 6,
        "end_idx": 8,
        "cards": ["A/2", "K/3"]
      },
      "state_before": {
        "hand_size": 11,
        "table_set_size": 0,
        "table_owner": null,
        "consecutive_scouts": 0,
        "score": 0
      }
    }
  ],
  "final_state": {
    "hand_sizes": [0, 3, 2, 1],
    "scores": [29, 5, 3, 2],
    "table_set": ["Q/4", "J/5"]
  },
  "payoffs": [29.0, 5.0, 3.0, 2.0],
  "winner": 0,
  "game_summary": {
    "total_moves": 37,
    "play_moves": 17,
    "scout_moves": 20,
    "winner_payoff": 29.0,
    "final_hand_sizes": [0, 3, 2, 1],
    "final_scores": [29, 5, 3, 2]
  }
}
```

## Action Types

### Play Actions
```json
{
  "type": "play",
  "description": "Play cards at positions 6-7: A/2, K/3",
  "start_idx": 6,
  "end_idx": 8,
  "cards": ["A/2", "K/3"]
}
```

### Scout Actions
```json
{
  "type": "scout",
  "description": "Scout Q/4 (flipped) from front, insert at position 3",
  "from_front": true,
  "insertion_in_hand": 3,
  "flip": true,
  "card_scouted": "Q/4"
}
```

## LLM Analysis Features

The log format is designed for easy LLM analysis:

1. **Structured Data**: JSON format with clear hierarchy
2. **Human-Readable Actions**: Decoded action descriptions
3. **Game Context**: State information before each move
4. **Game Statistics**: Summary data for pattern analysis
5. **Card Details**: Card values in format "top/bottom"
6. **Player Tracking**: Agent names and player IDs

## Example Analysis Queries

- "Which agent makes the most scout moves?"
- "What's the average game length for DQN vs Random?"
- "How often does the DQN agent play single cards vs sets?"
- "What's the win rate of DQN against Random agents?"
- "Which agent tends to scout from the front vs back of the table set?"

## File Organization

Logs are saved in the `logs/` directory with timestamps:
```
logs/
├── agent_games_20250623_165509.log
├── agent_games_20250623_170000.log
└── custom_games.log
```

## Requirements

- Trained DQN model at `experiments/scout_dqn_restart/model.pth`
- RLCard environment with Scout game
- Python 3.7+
