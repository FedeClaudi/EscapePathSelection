from dataclasses import dataclass
import numpy as np

@dataclass
class Action:
    idx: int # action index
    shift: np.ndarray  # position delta
    name: str

class Actions:
    LEFT = Action(0, np.array([-1, 0]), 'LEFT')
    RIGHT = Action(1, np.array([1, 0]), 'RIGHT')
    UP = Action(2, np.array([0, -1]), 'UP')
    DOWN = Action(3, np.array([0, 1]), 'DOWN')
    LEFT_UP = Action(4, np.array([-1, -1]), 'LEFT_UP')
    RIGHT_UP = Action(5, np.array([1, -1]), 'RIGHT_UP')
    RIGHT_DOWN = Action(6, np.array([1, 1]), 'RIGHT_DOWN')
    LEFT_DOWN = Action(7, np.array([-1, 1]), 'LEFT_DOWN')

    def __init__(self):
        self.actions = [self.LEFT, self.RIGHT, self.UP, self.DOWN,
                    self.LEFT_UP, self.RIGHT_UP, self.RIGHT_DOWN, self.LEFT_DOWN]

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, item):
        if isinstance(item, Action):
            return self.actions[item.idx]
        return self.actions[item]

    @property
    def lookup(self):
        return {a.name:a for a in self.actions}

    @classmethod
    def given_delta(self, delta):
        '''
            returns the action corresponding to a given position delta
        '''
        delta = np.array(delta)
        if delta.max()>1 or delta.min()<-1:
            raise ValueError('Position delta has values too large')

        actions = Actions().actions
        correct = [a for a in actions if np.all(a.shift == delta)]

        if not correct:
            raise ValueError(f'Could not match any action to delta: {delta}')
        else:
            return correct[0]