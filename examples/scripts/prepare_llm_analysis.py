#!/usr/bin/env python3
"""
Prepare Scout game logs for LLM analysis
"""

import json
import argparse
import os
from datetime import datetime

def load_game_logs(log_file):
    """Load and parse game logs"""
    games = []
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Split by double newlines to separate games
        game_blocks = content.strip().split('\n\n')
        
        for block in game_blocks:
            if block.strip():
                try:
                    game = json.loads(block)
                    games.append(game)
                except json.JSONDecodeError:
                    continue
    
    return games

def create_analysis_prompt(analysis_type="comprehensive"):
    """Create analysis prompt based on type"""
    
    base_prompt = """You are an expert game analyst specializing in the Scout card game and reinforcement learning agent behavior. You have been given detailed game logs from matches between different AI agents playing Scout.

## Game Context

**Scout** is a strategic card game where players:
- Start with hands of cards (each card has a top and bottom value, e.g., "A/2" means Ace on top, 2 on bottom)
- Can either PLAY cards (forming consecutive runs or same-value groups) or SCOUT cards (taking cards from the table)
- Score points by playing stronger sets than what's on the table
- Can flip cards to use their bottom value instead of top
- Win by having the highest score at the end

**Key Actions:**
- **PLAY**: Play cards from hand to beat the current table set
- **SCOUT**: Take a card from the table (front or back) and add it to your hand
- **FLIP**: When scouting, optionally flip the card to use its bottom value

## Analysis Instructions

Analyze the provided game logs to understand agent behavior patterns, strategies, and performance. Focus on:

### 1. **Agent Performance Analysis**
- Win rates and average payoffs
- Game length patterns (total moves, plays vs scouts)
- Final hand sizes and scoring efficiency

### 2. **Strategic Behavior Patterns**
- Play vs Scout decision making
- Card selection strategies (single cards vs sets)
- Front vs back scouting preferences
- Card flipping frequency and timing

### 3. **Learning and Adaptation**
- How different agent types (DQN vs Random) approach the game
- Evidence of strategic thinking vs random play
- Adaptation to opponent behavior

### 4. **Game Flow Analysis**
- Opening strategies
- Mid-game decision patterns
- End-game tactics
- Response to different game states

## Output Format

Please provide your analysis in the following structure:

### Executive Summary
- Key findings about agent performance and behavior
- Most interesting patterns discovered
- Overall assessment of agent strategies

### Detailed Analysis
- **Performance Metrics**: Win rates, game lengths, scoring patterns
- **Strategic Patterns**: Play vs scout decisions, card selection, scouting preferences
- **Agent Comparisons**: Differences between DQN and Random agents
- **Game Flow Insights**: Opening, mid-game, and end-game patterns

### Recommendations
- Suggestions for improving agent performance
- Potential strategy optimizations
- Areas for further investigation

### Data Insights
- Specific examples from the logs that illustrate key patterns
- Statistical summaries where relevant
- Notable game sequences or decisions

---

**Please analyze the following game logs and provide insights according to the above framework:**

"""
    
    if analysis_type == "performance":
        base_prompt += """
**FOCUS AREA**: Performance Analysis
Please emphasize:
- Win rates and scoring efficiency
- Game length optimization
- Performance differences between agent types
- Key success factors
"""
    elif analysis_type == "strategy":
        base_prompt += """
**FOCUS AREA**: Strategic Behavior Analysis
Please emphasize:
- Play vs scout decision patterns
- Card selection strategies
- Scouting preferences (front vs back, flipping)
- Strategic adaptation to game state
"""
    elif analysis_type == "comparison":
        base_prompt += """
**FOCUS AREA**: Agent Comparison Analysis
Please emphasize:
- DQN vs Random agent differences
- Evidence of learning and strategic thinking
- Behavioral patterns unique to each agent type
- Performance advantages and disadvantages
"""
    
    return base_prompt

def format_games_for_llm(games, max_games=None):
    """Format games for LLM consumption"""
    if max_games:
        games = games[:max_games]
    
    formatted_games = []
    for game in games:
        # Create a summary of the game
        game_summary = {
            "game_id": game["game_id"],
            "agents": game["agents"],
            "winner": game["winner"],
            "winner_agent": game["agents"][game["winner"]],
            "winner_payoff": game["payoffs"][game["winner"]],
            "total_moves": game["game_summary"]["total_moves"],
            "play_moves": game["game_summary"]["play_moves"],
            "scout_moves": game["game_summary"]["scout_moves"],
            "final_hand_sizes": game["final_state"]["hand_sizes"],
            "final_scores": game["final_state"]["scores"]
        }
        
        # Add sample moves (first 5 and last 5)
        moves = game["moves"]
        sample_moves = moves[:5] + moves[-5:] if len(moves) > 10 else moves
        
        game_summary["sample_moves"] = sample_moves
        formatted_games.append(game_summary)
    
    return formatted_games

def create_analysis_file(log_file, output_file, analysis_type="comprehensive", max_games=None):
    """Create complete analysis file with prompt and data"""
    
    # Load games
    games = load_game_logs(log_file)
    print(f"Loaded {len(games)} games from {log_file}")
    
    # Create prompt
    prompt = create_analysis_prompt(analysis_type)
    
    # Format games for LLM
    formatted_games = format_games_for_llm(games, max_games)
    
    # Create output
    output_content = prompt + "\n\n"
    output_content += f"## Game Logs ({len(formatted_games)} games)\n\n"
    output_content += json.dumps(formatted_games, indent=2, ensure_ascii=False, default=str)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"Analysis file created: {output_file}")
    print(f"Games included: {len(formatted_games)}")
    print(f"Analysis type: {analysis_type}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Prepare Scout game logs for LLM analysis")
    parser.add_argument("--log_file", type=str, required=True, help="Input log file path")
    parser.add_argument("--output_file", type=str, help="Output analysis file path")
    parser.add_argument("--analysis_type", type=str, default="comprehensive", 
                       choices=["comprehensive", "performance", "strategy", "comparison"],
                       help="Type of analysis to focus on")
    parser.add_argument("--max_games", type=int, help="Maximum number of games to include")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"analysis_{args.analysis_type}_{timestamp}.txt"
    
    create_analysis_file(args.log_file, args.output_file, args.analysis_type, args.max_games)

if __name__ == "__main__":
    main()
