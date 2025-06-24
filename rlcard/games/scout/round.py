import numpy as np
from copy import deepcopy
from typing import List
from .player import ScoutPlayer
from .judger import ScoutJudger 
from .dealer import ScoutDealer 
from .card import ScoutCard as Card
from .utils import segment_strength_rank, is_valid_scout_segment, find_all_scout_segments, get_action_list
from .utils.action_event import ScoutEvent, ScoutAction, PlayAction

DEBUG = False

class ScoutRound:
    def __init__(self, dealer: ScoutDealer, num_players: int):
        self.dealer = dealer
        self.num_players = num_players
        self.players = [ScoutPlayer(player_id=i) for i in range(num_players)]
        self.table_set: list[Card] = []
        self.table_owner: int = None
        self.consecutive_scouts = 0
        self.game_over = False

        # TODO: Cleaner
        n_cards = 45

        # Deal cards to each player
        self.dealer.shuffle()
        for i in range(num_players):
            initial_cards = self.dealer.deal_cards(n_cards // self.num_players)  # Implementation depends on your dealer
            self.players[i].hand = initial_cards

        # Choose first player
        self.current_player_id = 0  # or random if you prefer
        # self.payoff = 

    def proceed_round(self, action: ScoutEvent):
        """
        Process a player's action: either 'play' or 'scout'.
        Action structure might be something like:
            {
                'type': 'play',
                'cards': [hand positions],
            }
          or
            {
                'type': 'scout',
                'from_front': bool,
                'insertion_position_in_hand': int
            }
        Returns: next_player_id, rewards, done
        """
        player = self.players[self.current_player_id]

        if isinstance(action, PlayAction):
            # Validate the set is stronger than self.table_set
            # and that it's a consecutive run or same-value group in player's hand ordering
            valid = self._validate_play(player, action.start_idx, action.end_idx)
            if not valid:
                # Typically, RLCard might raise an error or return an invalid move penalty
                raise Exception("Invalid play action")
            
            # Execute the play
            new_set = player.play_cards(action.start_idx, action.end_idx)

            # Award points for beating the current table set
            if self.table_set:
                # Player gets points equal to the number of cards they beat
                player.score += len(self.table_set)

            ## TODO: Should be deque
            self.table_set = new_set
            self.table_owner = self.current_player_id
            self.consecutive_scouts = 0  # reset since we have a new table set

            # Possibly award a token or keep track of that for end-of-round scoring
            
            # Check if player emptied their hand
            if len(player.hand) == 0:
                self.game_over = True

        elif isinstance(action, ScoutAction):
            from_front = action.from_front
            insertion_position_in_hand = action.insertion_in_hand
            flip = action.flip

            # Remove card from table set
            scout_card = self.table_set.pop(0) if from_front else self.table_set.pop()
            
            # Flip the card if requested
            if flip:
                scout_card = scout_card.flip()
                
            player.insert_card(scout_card, insertion_position_in_hand)

            # The table owner gets +1 if the current player was forced to scout 
            # (i.e. they had no valid play). 
            # For RL simplicity, you might handle it here or in Judger.
            self.players[self.table_owner].score += 1

            self.consecutive_scouts += 1
            # If table_set is empty, might consider clearing table_owner 
            # or allow next player to effectively see an empty table set (which is easy to beat).

            # If consecutive_scouts == num_players, round ends
            if self.consecutive_scouts == self.num_players - 1:
                if DEBUG:
                    print( "All players have scouted consecutively, round ends.")
                self.game_over = True

        else:
            raise Exception("Unknown action type")

        # Prepare return values
        # rewards = self._compute_rewards() if self.game_over else [0]*self.num_players
        
        # Move to next player
        next_player_id = (self.current_player_id + 1) % self.num_players
        self.current_player_id = next_player_id

        if not len(self.get_legal_actions()):
            if DEBUG:
                print("No legal actions left, round ends.")
            self.game_over = True

        return next_player_id, self.game_over

    def _validate_play(self, player: ScoutPlayer, start_idx: int, end_idx: int):
        """
        Check if the chosen set is valid:
         1) Cards must be consecutive in player's hand 
            (or form a same-value group if that's how you're representing it).
         2) Must be stronger than self.table_set (longer or higher if same length).
        """
        # Retrieve the chosen cards
        if end_idx < start_idx or end_idx > len(player.hand) or start_idx < 0:
            print(f"Invalid Play. {start_idx=}{end_idx=}, but {len(player.hand)=}")
            return False

        chosen_cards = player.hand[start_idx:end_idx]

        # Check consecutive run or same-value group. 
        # Implementation depends on your representation of cards.
        if not is_valid_scout_segment(chosen_cards):
            return False

        # Compare strength with self.table_set
        if not self._is_stronger_set(chosen_cards, self.table_set):
            return False

        return True

    def get_payoffs(self):
        """
        Called when round ends. 
        Typically, you handle:
         - Hand penalty
         - Scout tokens 
         - Bonus for going out
        Return a list of rewards for each player.
        """
        # e.g. ask self.judger to do final scoring
        # from the partial info in this round
        return ScoutJudger.compute_rewards(self.players)
        # return ScoutJudger.judge_game(self.players, self.table_owner)

    def get_state(self, player_id: int):
        """
        RLCard typically requires a method to get the state
        from the perspective of a given player_id
        (masking hidden information as necessary).
        """
        player = self.players[player_id]
        # Build observation of:
        #  - the player's hand
        #  - the size of other players' hands
        #  - the table set (maybe only the front values are known if you hide back values)
        #  - any public scoring info
        # etc.
        state = {
            'hand': deepcopy(player.hand),
            'points': player.score,
            'table_set': deepcopy(self.table_set),
            'table_owner': self.table_owner,
            'consecutive_scouts': self.consecutive_scouts,
            'legal_actions': self.get_legal_actions(),
            'num_players': self.num_players,
            'current_player': self.current_player_id,
            'num_cards': {i: len(self.players[i].hand) for i in range(self.num_players)}
        }
        return state

    def get_legal_actions(self) -> List[ScoutEvent]:
        """
        Return a list of all possible actions from the perspective
        of the current player, i.e. all sets that can be played + all scout actions.
        """
        player = self.players[self.current_player_id]
        all_actions: List[ScoutEvent] = []

        # Generate the action list for the current hand size
        action_list = get_action_list(len(player.hand))

        # 1) Generate all valid sets the player could play
        possible_sets = find_all_scout_segments(player.hand)
        for s in possible_sets:
            if not self.table_set or self._is_stronger_set(s['cards'], self.table_set):
                all_actions.append(PlayAction(s['start'], s['end'], action_list))

        # 2) Scout actions
        # For each card in table_set, for each insertion position in player's hand
        if len(self.players[self.current_player_id].hand) < 15 and len(self.table_set) > 0:
            for insert_pos in range(len(player.hand) + 1):  # +1 to include inserting at the end
                # Add normal scout actions
                all_actions.append(ScoutAction(True, insert_pos, False, action_list))
                all_actions.append(ScoutAction(True, insert_pos, True, action_list))
                if len(self.table_set) > 1:
                    all_actions.append(ScoutAction(False, insert_pos, False, action_list))
                    all_actions.append(ScoutAction(False, insert_pos, True, action_list))

        return all_actions

    def _is_stronger_set(self, chosen_cards: list[Card], current_set: list[Card]):
        """
        Scout rule: A set is stronger than the current table set if:
        1) It has more cards than the table set, OR
        2) If it has the same number of cards, it follows Scout hierarchy:
           - Groups (matching cards) beat runs (consecutive cards) of same length
           - Within same type, higher rank wins
        If current_set is empty, anything is automatically 'stronger' (i.e. you can freely play).
        """
        # If table is empty, any set is valid to beat it
        if not current_set:
            return True

        new_len = len(chosen_cards)
        old_len = len(current_set)

        # 1) Compare lengths
        if new_len > old_len:
            return True
        elif new_len < old_len:
            return False

        # 2) Same length => compare using Scout hierarchy
        new_strength = segment_strength_rank(chosen_cards)
        old_strength = segment_strength_rank(current_set)

        # Compare type priority first (groups > runs > singles)
        if new_strength[0] > old_strength[0]:
            return True
        elif new_strength[0] < old_strength[0]:
            return False
        
        # Same type, compare rank
        return new_strength[1] > old_strength[1]

