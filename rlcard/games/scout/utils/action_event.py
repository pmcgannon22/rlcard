from .utils import get_action_list

ACTION_LIST = get_action_list()
ACTION_FROM_ID = {v: k for k, v in ACTION_LIST.items()}
print(ACTION_LIST)

class ScoutEvent:
    def __init__(self, action_id):
        self.action_id = action_id

    def get_action_repr(self):
        raise NotImplementedError
        
    @staticmethod
    def from_action_id(action_id: int):
        action_str: str = ACTION_FROM_ID[action_id]
        vals = action_str.split('-')
        if vals[0] == 'play':
            return PlayAction(int(vals[1]), int(vals[2]))
        elif vals[0] == 'scout':
            return ScoutAction(vals[1] == 'front', int(vals[2]))
        else:
            raise Exception(f"Do not recognize {action_id=}")
        
    def __str__(self):
        return self.get_action_repr()
    
    def __repr__(self):
        return self.get_action_repr()

class ScoutAction(ScoutEvent):
    def __init__(self, from_front: bool, insertion_in_hand: int):
        self.from_front = from_front
        self.insertion_in_hand = insertion_in_hand
        super().__init__(ACTION_LIST[self.get_action_repr()])
    
    def get_action_repr(self) -> str:
        return f"scout-{'front' if self.from_front else 'back'}-{self.insertion_in_hand}"

    def __str__(self) -> str:
        return self.get_action_repr()

    def __repr__(self):
        return self.get_action_repr()

class PlayAction(ScoutEvent):
    def __init__(self, start_idx: int, end_idx: int):
        self.start_idx = start_idx
        self.end_idx = end_idx
        super().__init__(ACTION_LIST[self.get_action_repr()])

    def get_action_repr(self) -> str:
        return f"play-{self.start_idx}-{self.end_idx}"

    def __str__(self):
        return self.get_action_repr()
    
    def __repr__(self):
        return self.get_action_repr()