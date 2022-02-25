import numpy as np
import random

from dataclasses import dataclass

from rl.environment.actions import Action
from rl.models.Q import QLearner
from rl.models.tracking import QLearnerTracking

@dataclass
class ModelEntry:
    action: int
    reward: float
    next_state: tuple

class Model:
    '''
        It's a dictionary with (state) as key
        and entries mapping each action to 
        a reward and the next state.
    '''
    def __init__(self, n_actions):
        self.model = {}
        self.n_actions = n_actions

    def __getitem__(self, item):
        return self.model[item]

    @property
    def entries(self):
        return list(self.model.keys())

    def _add_entry(self, state):
        self.model[state] = {a: None for a in range(self.n_actions)}

    def update(self, state, action, next_state, reward):
        if state not in self.entries:
            self._add_entry(state)
        
        # update entry
        self.model[state][action] = ModelEntry(action, reward, next_state)


class DynaPlanner:
    def plan(self):
        '''
            Draws N random states stored in the model, selects a random
            action performed at that state stored in the Model and sees
            what the reward/next state were. Uses all of this to update
            the Q table.
        '''
        for step in range(self.n_planning_steps):
            # select a model entry
            state = random.choice(self.model.entries)

            # select one of previously executed actions
            _actions = [action for action, entry in self.model[state].items() if entry is not None]
            action = random.choice(_actions)
            entry = self.model[state][action]
            
            # compute expected value
            if entry.next_state == -1:
                expect = 0
            else:
                next_state_index = self.state_indices[entry.next_state]
                expect = self.discount * np.max(self.Q[next_state_index, :])

            # update Q table
            state_index = self.state_indices[state]
            self.Q[state_index, action] += self.learning_rate * (entry.reward + expect - self.Q[state_index,action])



# ----------------------------------- DYNAQ ---------------------------------- #
class DynaQModel(QLearner, DynaPlanner):
    def __init__(self, game, name='DynaQ', *args, n_planning_steps=30, **kwargs):
        super().__init__(game, name='DynaQ', **kwargs)
        DynaPlanner.__init__(self)
        self.n_planning_steps = n_planning_steps
        
        self.model = Model(self.environment.n_actions)

        # add params
        self.params['n_planning_steps'] = self.n_planning_steps

    def training_step(self, state_index):
        '''
            Single step during training of one episode.

            Arguments:
                state_index: int. State index
        '''
        # select an action
        action = self.choose_epsilon_greedy_action(state_index)

        # execute action
        next_state, reward, status = self.environment.step(action)
        next_state_index = self.state_indices[next_state]
    
        # check everything went okay
        if isinstance(action, Action):
            action = action.idx

        _action = self.environment.actions[action]
        if next_state_index == state_index and action in self.environment._possible_actions():
            raise ValueError(f'Action: {action} didnt change the state!')
        
        if _action in self.environment._possible_actions() and reward < 0:
            raise ValueError('How can a possible action have negative reward?')

        # update Q table
        max_next_Q = self.state_q_max(state_index)
        self.Q[state_index, action] += self.learning_rate * (reward + self.discount * max_next_Q - self.Q[state_index, action])

        # print(f"""
        #     {self.state_lookup[state_index]}
        #     reward: {reward}
        #     discounted expected: {self.discount * max_next_Q}
        #     delta: {(reward + self.discount * max_next_Q - self.Q[state_index, action])}
        #     q: {self.Q[state_index, action]}
        
        # """)

        # update model and plan
        self.model.update(self.state_lookup[state_index], action, next_state, reward)
        self.plan()

        return next_state_index, reward, status

# ------------------------------ tracking DYNAQ ------------------------------ #
class DynaQTracking(QLearnerTracking, DynaPlanner):
    def __init__(self, environment, maze_name, trial_number=None, name='DynaQTracking', n_planning_steps=150, **kwargs):
        super().__init__(environment, maze_name, trial_number=trial_number, name='DynaQTracking', **kwargs)
        DynaPlanner.__init__(self)
        self.n_planning_steps = n_planning_steps

        self.model = Model(self.environment.n_actions)
        self.params['n_planning_steps'] = self.n_planning_steps

    def tracking_training_step(self, state_index, action, next_state_index, reward):
        # update Q based on the action the mouse took and the reward experienced
        max_next_Q = self.state_q_max(state_index)
        self.Q[state_index, action.idx] += self.learning_rate * (reward + self.discount * max_next_Q - self.Q[state_index, action.idx])

        # update model and plan
        self.model.update(self.state_lookup[state_index], action.idx, self.state_lookup[next_state_index], reward)
        self.plan()