#!/bin/bash

# Simple script to play Scout against DQN AI
echo "🎮 Starting Scout Game - Human vs DQN AI"
echo "========================================"

python3 -c "
import torch
import rlcard
from rlcard.agents.human_agents.scout_human_agent import HumanAgent

# Load the trained DQN model
device = 'cpu'
dqn_agent = torch.load('experiments/scout_dqn_restart/model.pth', weights_only=False, map_location=device)
dqn_agent.set_device(device)

# Create human agent
human_agent = HumanAgent(num_actions=rlcard.make('scout').num_actions)

# Set up the game
env = rlcard.make('scout')
env.set_agents([human_agent, dqn_agent, dqn_agent, dqn_agent])

print('>> Scout Game - Human vs AI')
print('>> You are Player 0')
print('>> Good luck!')
print('')

while True:
    print('>> Start a new game')
    print('=' * 60)
    
    trajectories, payoffs = env.run(is_training=False)
    
    print('=============== Result ===============')
    if payoffs[0] > 0:
        print('🎉 You win!')
    elif payoffs[0] == 0:
        print('🤝 It is a tie.')
    else:
        print('😔 You lose!')
    
    print(f'Your payoff: {payoffs[0]}')
    print('')
    
    try:
        user_input = input('Press Enter to continue or q to quit: ')
        if user_input.lower() == 'q':
            print('Thanks for playing!')
            break
    except KeyboardInterrupt:
        print('\\nThanks for playing!')
        break
"
