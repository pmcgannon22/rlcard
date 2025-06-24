#!/bin/bash

# Scout Game - Human vs DQN AI
# This script allows you to play Scout against your trained DQN model

echo "🎮 Scout Game - Human vs DQN AI"
echo "=================================="
echo "You will play against 3 trained DQN agents"
echo "Good luck!"
echo ""

# Check if model exists
MODEL_PATH="experiments/scout_dqn_restart/model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: DQN model not found at $MODEL_PATH"
    echo "Please train a model first using:"
    echo "python3 examples/run_rl.py --env scout --algorithm dqn --num_episodes 15000 --log_dir experiments/scout_dqn_restart/"
    exit 1
fi

echo "✅ Found DQN model at $MODEL_PATH"
echo ""

# Run the human vs AI game
python3 -c "
import torch
import rlcard
from rlcard.agents.human_agents.scout_human_agent import HumanAgent

# Load the trained DQN model
device = 'cpu'
dqn_agent = torch.load('$MODEL_PATH', weights_only=False, map_location=device)
dqn_agent.set_device(device)

# Create human agent
human_agent = HumanAgent(num_actions=rlcard.make('scout').num_actions)

# Set up the game with human player and AI opponents
env = rlcard.make('scout')
env.set_agents([
    human_agent,      # Human player (Player 0)
    dqn_agent,        # AI opponent (Player 1)
    dqn_agent,        # AI opponent (Player 2)
    dqn_agent,        # AI opponent (Player 3)
])

print('>> Scout Game - Human vs AI')
print('>> You are Player 0')
print('>> You will play against 3 trained DQN agents')
print('>> Good luck!')
print('')

while True:
    print('>> Start a new game')
    print('=' * 60)
    
    trajectories, payoffs = env.run(is_training=False)
    
    # Show final game state
    print('=============== Final Game State ===============')
    perfect_info = env.get_perfect_information()
    for i in range(env.num_players):
        player_hand = perfect_info.get('hand_cards', [])[i] if i < len(perfect_info.get('hand_cards', [])) else []
        print(f'Player {i} hand: {player_hand}')
    print('')
    
    print('=============== Result ===============')
    if payoffs[0] > 0:
        print('🎉 You win!')
    elif payoffs[0] == 0:
        print('🤝 It is a tie.')
    else:
        print('😔 You lose!')
    
    print(f'Your payoff: {payoffs[0]}')
    print('')
    
    # Ask if user wants to continue
    try:
        user_input = input('Press Enter to continue or q to quit: ')
        if user_input.lower() == 'q':
            print('Thanks for playing!')
            break
    except KeyboardInterrupt:
        print('\\nThanks for playing!')
        break
"
