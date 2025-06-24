from .utils import get_action_list

class ScoutEvent:
    def __init__(self, action_id, action_list):
        self.action_id = action_id
        self.action_list = action_list

    def get_action_repr(self):
        # Reverse lookup
        for k, v in self.action_list.items():
            if v == self.action_id:
                return k
        return str(self.action_id)
        
    @staticmethod
    def from_action_id(action_id, action_list):
        # Use the provided action_list for reverse lookup
        for k, v in action_list.items():
            if v == action_id:
                action_str = k
                break
        else:
            raise Exception(f"Do not recognize {action_id=}")
        vals = action_str.split('-')
        if vals[0] == 'play':
            return PlayAction(int(vals[1]), int(vals[2]), action_list)
        elif vals[0] == 'scout':
            return ScoutAction(vals[1] == 'front', int(vals[2]), vals[3] == 'flip', action_list)
        else:
            raise Exception(f"Do not recognize {action_id=}")
        
    def __str__(self):
        return self.get_action_repr()
    
    def __repr__(self):
        return self.get_action_repr()

class ScoutAction(ScoutEvent):
    def __init__(self, from_front: bool, insertion_in_hand: int, flip: bool = False, action_list=None):
        self.from_front = from_front
        self.insertion_in_hand = insertion_in_hand
        self.flip = flip
        if action_list is None:
            raise ValueError("action_list must be provided")
        super().__init__(action_list[self.get_action_repr()], action_list)
    
    def get_action_repr(self) -> str:
        flip_suffix = '-flip' if self.flip else '-normal'
        return f"scout-{'front' if self.from_front else 'back'}-{self.insertion_in_hand}{flip_suffix}"

    def __str__(self) -> str:
        return self.get_action_repr()

    def __repr__(self):
        return self.get_action_repr()

class PlayAction(ScoutEvent):
    def __init__(self, start_idx: int, end_idx: int, action_list=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        if action_list is None:
            raise ValueError("action_list must be provided")
        super().__init__(action_list[self.get_action_repr()], action_list)

    def get_action_repr(self) -> str:
        return f"play-{self.start_idx}-{self.end_idx}"

    def __str__(self):
        return self.get_action_repr()
    
    def __repr__(self):
        return self.get_action_repr()