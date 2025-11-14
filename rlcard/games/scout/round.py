from __future__ import annotations

from copy import deepcopy
from typing import Any

from .player import ScoutPlayer
from .judger import ScoutJudger 
from .dealer import ScoutDealer 
from .card import ScoutCard as Card
from .utils import (
    CardSegment,
    find_all_scout_segments,
    get_action_list,
    is_valid_scout_segment,
    segment_strength_rank,
)
from .utils.action_event import ScoutEvent, ScoutAction, PlayAction

DEBUG = False

class ScoutRound:
    def __init__(
        self,
        dealer: ScoutDealer,
        num_players: int,
        force_play_if_possible: bool = True,
        max_hand_size: int = 16,
    ) -> None:
        self.dealer = dealer
        self.num_players = num_players
        self.force_play_if_possible = force_play_if_possible
        self.max_hand_size = max_hand_size
        self.players: list[ScoutPlayer] = [ScoutPlayer(player_id=i) for i in range(num_players)]
        self.table_set: list[Card] = []
        self.table_owner: int | None = None
        self.consecutive_scouts: int = 0
        self.game_over: bool = False
        # Deal cards in a round-robin fashion so ordering stays meaningful
        self.dealer.shuffle()
        player_idx = 0
        while self.dealer.deck:
            card = self.dealer.deal_cards(1)[0]
            self.players[player_idx].hand.append(card)
            player_idx = (player_idx + 1) % num_players

        # Choose first player
        self.current_player_id: int = self.dealer.np_random.randint(self.num_players)

    def proceed_round(self, action: ScoutEvent) -> tuple[int, bool]:
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
            if not self.table_set:
                raise RuntimeError("Cannot scout from an empty table set.")
            scout_card = self.table_set.pop(0) if from_front else self.table_set.pop()
            
            # Flip the card if requested
            if flip:
                scout_card = scout_card.flip()
                
            player.insert_card(scout_card, insertion_position_in_hand)

            # The table owner gets +1 if the current player was forced to scout 
            # (i.e. they had no valid play). 
            # For RL simplicity, you might handle it here or in Judger.
            if self.table_owner is not None:
                self.players[self.table_owner].score += 1

            self.consecutive_scouts += 1
            if not self.table_set:
                # Empty table => next player can freely play anything
                self.table_owner = None
                self.consecutive_scouts = 0

            if self.consecutive_scouts == self.num_players - 1 and self.table_owner is not None:
                if DEBUG:
                    print("All players have scouted consecutively, round ends.")
                self.players[self.table_owner].score += len(self.table_set)
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

    def _validate_play(self, player: ScoutPlayer, start_idx: int, end_idx: int) -> bool:
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

    def get_payoffs(self) -> list[int]:
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

    def get_state(self, player_id: int) -> dict[str, Any]:
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
        state: dict[str, Any] = {
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

    def get_legal_actions(self) -> list[ScoutEvent]:
        """
        Return a list of all possible actions from the perspective
        of the current player, i.e. all sets that can be played + all scout actions.
        """
        player = self.players[self.current_player_id]
        all_actions: list[ScoutEvent] = []

        # Generate the action list for the maximum hand size to keep IDs consistent
        action_list = get_action_list(self.max_hand_size)

        # 1) Generate all valid sets the player could play
        possible_sets: list[CardSegment] = find_all_scout_segments(player.hand)
        playable_segments: list[CardSegment] = []
        for segment in possible_sets:
            if not self.table_set or self._is_stronger_set(segment['cards'], self.table_set):
                playable_segments.append(segment)

        for segment in playable_segments:
            all_actions.append(PlayAction(segment['start'], segment['end'], action_list))

        # 2) Scout actions
        # For each card in table_set, for each insertion position in player's hand
        can_scout = len(self.table_set) > 0 and len(player.hand) < self.max_hand_size

        if can_scout:
            allow_back = len(self.table_set) > 1
            for insert_pos in range(len(player.hand) + 1):
                all_actions.append(ScoutAction(True, insert_pos, False, action_list))
                all_actions.append(ScoutAction(True, insert_pos, True, action_list))
                if allow_back:
                    all_actions.append(ScoutAction(False, insert_pos, False, action_list))
                    all_actions.append(ScoutAction(False, insert_pos, True, action_list))

        return all_actions

    def _is_stronger_set(self, chosen_cards: list[Card], current_set: list[Card]) -> bool:
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
