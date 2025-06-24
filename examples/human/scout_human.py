''' A toy example of playing Scout against AI agents
'''
from rlcard.agents import RandomAgent

import rlcard
from rlcard.agents.human_agents.scout_human_agent import HumanAgent

# Make environment
env = rlcard.make('scout')

# Create agents
human_agent = HumanAgent(env.num_actions)
random_agent = RandomAgent(num_actions=env.num_actions)

# Set up the game with human player and AI opponents
env.set_agents([
    human_agent,  # Human player (Player 0)
    random_agent, # AI opponent (Player 1)
    random_agent, # AI opponent (Player 2)
    random_agent, # AI opponent (Player 3)
])

print(">> Scout Game")
print(">> You are Player 0")
print(">> Rules: Play sets of cards (runs or groups) or scout cards from the table")
print(">> A set is stronger if it has more cards, or if same length, higher rank")
print(">> If you can't play, you must scout (take a card from table)")
print(">> First player to empty their hand wins!")
print("")

while True:
    print(">> Start a new game")
    print("=" * 60)

    trajectories, payoffs = env.run(is_training=False)
    
    # If the human does not take the final action, we need to
    # print other players' actions
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state.get('current_player', 0):
            break
        _action_list.insert(0, action_record[-i])
    
    if _action_list:
        print('\n=============== Final Actions ===============')
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses ', end='')
            if hasattr(pair[1], 'get_action_repr'):
                print(pair[1].get_action_repr())
            else:
                print(pair[1])
        print('')

    # Show final game state
    print('=============== Final Game State ===============')
    perfect_info = env.get_perfect_information()
    for i in range(env.num_players):
        player_hand = perfect_info.get('hand_cards', [])[i] if i < len(perfect_info.get('hand_cards', [])) else []
        print(f'Player {i} hand: {player_hand}')
    print('')

    print('=============== Result ===============')
    if payoffs[0] > 0:
        print('You win!')
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose!')
    
    print(f'Your payoff: {payoffs[0]}')
    print('')
    
    # Display scores for all players
    print('=============== Final Scores ===============')
    for i in range(env.num_players):
        player_score = env.game.round.players[i].score
        player_hand_size = len(env.game.round.players[i].hand)
        player_type = "You" if i == 0 else f"AI Player {i}"
        print(f'{player_type}: {player_score} points (hand size: {player_hand_size})')
    print('')

    # Ask if user wants to continue
    try:
        user_input = input("Press Enter to continue or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Thanks for playing!")
            break
    except KeyboardInterrupt:
        print("\nThanks for playing!")
        break 