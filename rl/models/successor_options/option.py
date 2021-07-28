from dataclasses import dataclass
import pickle
import numpy as np

@dataclass
class Option:
    number: int
    subgoal_state: tuple  # state corresponding to option's subgoal
    subgoal_state_index: int  # index of subgoal state
    Q: np.array

    def __repr__(self):
        return f'Option: {self.number} to subgoal: {self.subgoal_state} | initiation set: {len(self.initiation_set)} | termination set: {len(self.termination_set)}'
    
    def __str__(self):
        return self.__repr__()

    def save(self):
        with open(f'./data/option_{self.number}.pkl', 'wb') as out:
            pickle.dump(self, out)
    
    @classmethod
    def load(cls, number):
        with open(f'./data/option_{number}.pkl', 'rb') as fin:
            return pickle.load(fin)

    @property
    def initiation_set(self):
        '''
            Returns the set of states indices in which the option can be initiated.
            That's defined as the states in which the option's Q is >0 for at least one action
        '''
        return np.argwhere(self.Q.max(1) > 0).ravel()
    
    @property
    def termination_set(self):
        '''
            Returns the option's termination set which is defined as the set of states in which
            the option's policy is <= 0 for all actions
        '''
        return np.argwhere(self.Q.max(1) <= 0).ravel()

    def get_action(self, state_index):
        '''
            Returns the action with the higest value at a state
        '''
        qvals = self.Q[state_index, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]