import random
import numpy as np
from loguru import logger

from rl.environment.actions import Action
from rl.models.Q import QLearner
from rl.models.tracking import QLearnerTracking


class QTableModel(QLearner):
    def __init__(self, game, name='QTable', *args, **kwargs):
        super().__init__(game, name='QTable', **kwargs)
        # logger.info(f"Models parameters '{name}': {self.params}")


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

        # update Q table
        if isinstance(action, Action):
            action = action.idx

        max_next_Q = self.state_q_max(state_index)
        self.Q[state_index, action] += self.learning_rate * \
            (reward + self.discount * max_next_Q - self.Q[state_index, action])
        return next_state_index, reward, status


class QTableTracking(QLearnerTracking):
    '''
        Tabular Q-learning agent which uses real mouse tracking data for learning:
        it replicates the experience of a real mouse to see how a q-learning would do compared to the mouse
    '''
    def __init__(self, environment, maze_name, trial_number=None, name='QTableTracking', **kwargs):
        super().__init__(environment, maze_name, trial_number=trial_number, name='QTableTracking', **kwargs)


    def tracking_training_step(self, state_index, action, next_state_index, reward):
        # update Q based on the action the mouse took and the reward experienced
        max_next_Q = self.state_q_max(state_index)
        self.Q[state_index, action.idx] += self.learning_rate * (reward + self.discount * max_next_Q - self.Q[state_index, action.idx])
