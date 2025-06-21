from rlcard.games.scout.card import ScoutCard
from rlcard.games.scout.utils.action_event import ScoutEvent, PlayAction, ScoutAction

class HumanAgent(object):
    ''' A human agent for Scout. It can be used to play against trained models
    '''

    def __init__(self, num_actions):
        ''' Initialize the human agent

        Args:
            num_actions (int): the size of the output action space
        '''
        self.use_raw = True
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        _print_state(state['raw_obs'], state['action_record'])
        action = _get_human_action(state['raw_legal_actions'])
        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}

def _print_state(state, action_record):
    ''' Print out the state of a given player

    Args:
        state (dict): The current game state
        action_record (list): A list of the historical actions
    '''
    # Print recent actions
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state.get('current_player', 0):
            break
        _action_list.insert(0, action_record[-i])
    
    if _action_list:
        print('\n=============== Recent Actions ===============')
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses ', end='')
            _print_action(pair[1])
            print('')

    # Print current game state
    print('\n=============== Your Hand ===============')
    if state['hand']:
        print_hand_graphical(state['hand'])
        print(f'Hand size: {len(state["hand"])} cards')
    else:
        print('Empty hand')
    print('')
    
    print('=============== Table Set ===============')
    if state['table_set']:
        ScoutCard.print_cards(state['table_set'])
        if state.get('table_owner') is not None:
            print(f'Owned by Player {state["table_owner"]}')
    else:
        print('No cards on table')
    print('')
    
    print('=============== Game Info ===============')
    print(f'Your score: {state.get("points", 0)}')
    print(f'Consecutive scouts: {state.get("consecutive_scouts", 0)}')
    print('')

def print_hand_graphical(hand):
    '''
    Print the player's hand as a row of boxes with <top>/<bottom> and indices below.
    '''
    # Each card will be a box: ┌─────┐, │ 6/9 │, └─────┘
    top_line = ''
    mid_line = ''
    bot_line = ''
    idx_line = ''
    for idx, card in enumerate(hand):
        val = f'{card.top}/{card.bottom}'
        # Center the value in 5 spaces
        val = val.center(5)
        top_line += '┌─────┐ '
        mid_line += f'│{val}│ '
        bot_line += '└─────┘ '
        idx_str = str(idx).center(7)
        idx_line += idx_str
    print(top_line.rstrip())
    print(mid_line.rstrip())
    print(bot_line.rstrip())
    print(idx_line.rstrip())

def _print_action(action):
    ''' Print out an action in a nice form

    Args:
        action (ScoutEvent): A ScoutEvent action
    '''
    if isinstance(action, PlayAction):
        print(f'PLAY cards at positions {action.start_idx} to {action.end_idx-1}')
    elif isinstance(action, ScoutAction):
        direction = "front" if action.from_front else "back"
        print(f'SCOUT from {direction} of table set, insert at position {action.insertion_in_hand}')
    else:
        print(str(action))

def _get_human_action(legal_actions):
    ''' Get action from human input

    Args:
        legal_actions (list): A list of legal actions

    Returns:
        action (ScoutEvent): The action chosen by human
    '''
    print('=========== Actions You Can Choose ===========')
    
    # Group actions by type for better display
    play_actions = []
    scout_actions = []
    
    for i, action in enumerate(legal_actions):
        if isinstance(action, PlayAction):
            play_actions.append((i, action))
        elif isinstance(action, ScoutAction):
            scout_actions.append((i, action))
    
    # Display play actions
    if play_actions:
        print('\n--- PLAY Actions (play cards from your hand) ---')
        for idx, action in play_actions:
            print(f'{idx}: ', end='')
            _print_action(action)
    
    # Display scout actions
    if scout_actions:
        print('\n--- SCOUT Actions (take card from table) ---')
        for idx, action in scout_actions:
            print(f'{idx}: ', end='')
            _print_action(action)
    
    print('\n=============================================')
    
    # Get user input
    while True:
        try:
            action_idx = int(input('>> Choose action (enter number): '))
            if 0 <= action_idx < len(legal_actions):
                return legal_actions[action_idx]
            else:
                print(f'Invalid action number. Please choose between 0 and {len(legal_actions)-1}')
        except ValueError:
            print('Please enter a valid number')
        except KeyboardInterrupt:
            print('\nGame interrupted by user')
            exit(0)
