from rlcard.games.scout.card import ScoutCard
from rlcard.games.scout.utils.action_event import ScoutEvent, PlayAction, ScoutAction

class HumanAgent(object):
    ''' A human agent for Scout. It can be used to play against trained models
    '''

    def __init__(self, num_actions, advisor=None, suggestion_label="⭐"):
        ''' Initialize the human agent

        Args:
            num_actions (int): the size of the output action space
            advisor: Optional RL agent used to suggest moves
            suggestion_label (str): Marker displayed next to the suggested action
        '''
        self.use_raw = True
        self.num_actions = num_actions
        self.advisor = advisor
        self.suggestion_label = suggestion_label

    def step(self, state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        _print_state(state['raw_obs'], state['action_record'])
        action = _get_human_action(
            state['raw_legal_actions'],
            state,
            advisor=self.advisor,
            suggestion_label=self.suggestion_label,
        )
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
            _print_recent_action(pair[1])
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
    consecutive = state.get("consecutive_scouts", 0)
    num_players = state.get("num_players", len(state.get("num_cards", {})) or 4)
    warning = ""
    if state.get("table_owner") is not None and consecutive + 1 >= num_players - 1:
        warning = " \u26a0\ufe0f WARNING: Next scout ends round!"
    print(f'Consecutive scouts: {consecutive}{warning}')
    
    # Show other players' hand sizes if available
    if 'num_players' in state:
        print('\n========== Other Players ==========')
        for i in range(state['num_players']):
            if i != state.get('current_player', 0):
                hand_size = state.get('num_cards', {}).get(i, '?')
                print(f'Player {i}: {hand_size} cards')
    print('')

def _print_recent_action(action_info):
    ''' Print out a recent action with enhanced information

    Args:
        action_info: Either a ScoutEvent action or a tuple of (player_id, action, context)
    '''
    # Handle both old format (just action) and new format (action with context)
    if isinstance(action_info, tuple) and len(action_info) >= 3:
        player_id, action, context = action_info
        if context.get('action_type') == 'play':
            cards_str = ', '.join([f'[{card}]' for card in context['cards']])
            print(f'PLAY {cards_str}')
        elif context.get('action_type') == 'scout':
            card_str = f"[{context.get('card', 'Unknown')}]"
            direction = context['direction']
            flip_str = ' (flipped)' if context.get('flipped', False) else ''
            print(f'SCOUT {card_str}{flip_str} from {direction} of table set')
        else:
            # Fallback to basic action display
            if isinstance(action, PlayAction):
                print(f'PLAY cards at positions {action.start_idx} to {action.end_idx-1}')
            elif isinstance(action, ScoutAction):
                direction = "front" if action.from_front else "back"
                flip_str = ' (flipped)' if getattr(action, 'flip', False) else ''
                print(f'SCOUT from {direction} of table set, insert at position {action.insertion_in_hand}{flip_str}')
            else:
                print(str(action))
    else:
        # Handle old format (just action object)
        action = action_info
        if isinstance(action, PlayAction):
            print(f'PLAY cards at positions {action.start_idx} to {action.end_idx-1}')
        elif isinstance(action, ScoutAction):
            direction = "front" if action.from_front else "back"
            flip_str = ' (flipped)' if getattr(action, 'flip', False) else ''
            print(f'SCOUT from {direction} of table set, insert at position {action.insertion_in_hand}{flip_str}')
        else:
            print(str(action))

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
        # Center the index under the card box (6 characters wide: ┌─────┐ + space)
        idx_str = str(idx).center(8)
        idx_line += idx_str
    print(top_line)
    print(mid_line)
    print(bot_line)
    print(idx_line)

def _print_action(action, state=None):
    ''' Print out an action in a nice form

    Args:
        action (ScoutEvent): A ScoutEvent action
        state (dict): The current game state (optional, for enhanced display)
    '''
    if isinstance(action, PlayAction):
        if state and 'hand' in state:
            # Get the actual cards being played
            cards = state['hand'][action.start_idx:action.end_idx]
            card_strs = [f'[{card.top}/{card.bottom}]' for card in cards]
            cards_desc = ', '.join(card_strs)
            print(f'PLAY cards at positions {action.start_idx} to {action.end_idx-1} ({cards_desc})')
        else:
            print(f'PLAY cards at positions {action.start_idx} to {action.end_idx-1}')
    elif isinstance(action, ScoutAction):
        direction = "front" if action.from_front else "back"
        flip_str = ' (flipped)' if getattr(action, 'flip', False) else ''
        if state and 'table_set' in state and state['table_set']:
            # Get the card being scouted
            if action.from_front:
                scout_card = state['table_set'][0]
            else:
                scout_card = state['table_set'][-1]
            card_desc = f'[{scout_card.top}/{scout_card.bottom}]'
            # Describe insertion position
            if state and 'hand' in state:
                hand = state['hand']
                if action.insertion_in_hand == 0:
                    pos_desc = f"before [{hand[0].top}/{hand[0].bottom}]"
                elif action.insertion_in_hand >= len(hand):
                    pos_desc = f"after [{hand[-1].top}/{hand[-1].bottom}]"
                else:
                    pos_desc = f"between [{hand[action.insertion_in_hand-1].top}/{hand[action.insertion_in_hand-1].bottom}], [{hand[action.insertion_in_hand].top}/{hand[action.insertion_in_hand].bottom}]"
            else:
                pos_desc = f"at position {action.insertion_in_hand}"
            print(f'SCOUT {card_desc}{flip_str} from {direction} of table set, insert {pos_desc}')
        else:
            print(f'SCOUT from {direction} of table set, insert at position {action.insertion_in_hand}{flip_str}')
    else:
        print(str(action))

def _find_suggested_action(legal_actions, state, advisor):
    if advisor is None:
        return None
    try:
        suggested_id, _ = advisor.eval_step(state)
    except Exception:
        return None
    for action in legal_actions:
        if getattr(action, 'action_id', None) == suggested_id:
            return action
    return None

def _get_human_action(legal_actions, state, advisor=None, suggestion_label="⭐"):
    ''' Get action from human input

    Args:
        legal_actions (list): A list of legal actions
        state (dict): The current game state
        advisor: Optional RL agent for suggested move

    Returns:
        action (ScoutEvent): The action chosen by human
    '''
    print('=========== Actions You Can Choose ===========')
    raw_state = state['raw_obs']
    suggested_action = _find_suggested_action(legal_actions, state, advisor)
    
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
            marker = suggestion_label if action is suggested_action else ' '
            print(f'{idx}: {marker} ', end='')
            _print_action(action, raw_state)
    
    # Display scout actions
    if scout_actions:
        print('\n--- SCOUT Actions (take card from table) ---')
        for idx, action in scout_actions:
            marker = suggestion_label if action is suggested_action else ' '
            print(f'{idx}: {marker} ', end='')
            _print_action(action, raw_state)
    
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
