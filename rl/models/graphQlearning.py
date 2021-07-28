import numpy as np
import random

from rl.models.Q import QLearner
from rl.environment.environment import Actions

class GraphQModel(QLearner):
    def __init__(self, game, name='GraphQ', *args, **kwargs):
        super().__init__(game, name='GraphQ', **kwargs)

    def training_step(self, state_index):

        # select and execute action
        action = self.choose_epsilon_greedy_action(state_index)

        # execute action
        next_state, reward, status = self.environment.step(action)
        next_state_index = self.state_indices[next_state]

        # update Q table
        max_next_Q = self.state_q_max(state_index)
        self.Q[state_index, action] += self.learning_rate * \
            (reward + self.discount * max_next_Q - self.Q[state_index, action])

        return next_state_index, reward, status

    def _get_neighbors(self, state_index):
        '''
            Get states connected to state index on maze graph
        '''
        if not isinstance(state_index, tuple):
            state = self.state_lookup[state_index]
        else:
            state = state_index
        return list(self.environment.graph.neighbors(state))

    def _get_actions(self, state, *neighbors):
        '''
            Get the actions that take the agent from state to
            a variable number of state neighbors
        '''
        actions = []
        for nb in neighbors:
            delta = np.array(state) - np.array(nb)
            
            if delta.max() > 1 or delta.min() < -1 or np.abs(delta).max() == 0:
                raise ValueError(f'It appears that {state} and {nb} are not neighbors')

            if delta[0] == 1:
                actions.append(Action.MOVE_RIGHT)
            elif delta[0] == -1:
                actions.append(Action.MOVE_LEFT)
            elif delta[1] == 1:
                actions.append(Action.MOVE_UP)
            else:
                actions.append(Action.MOVE_DOWN)

        return actions

    def get_graph_action(self, state_index):
        '''
            Select an action with epsilon greedy policy
        '''
        # choose action epsilon greedy (off-policy, instead of only using the learned policy)
        if np.random.random() < self.exploration_rate and not self.deterministic:
            # get node neighbors from the graph
            # neighbors = self._get_neighbors(state_index)
            # actions = self._get_actions(self.state_lookup[state_index], *neighbors)
            actions = self.environment._possible_actions(self.state_lookup[state_index])
            action = random.choice(actions)
        else:
            action = self.predict(state_index)

        return action

    def predict(self, state_index=0):
        '''
            Get the best action given the current state
        '''
        # get available actions
        if isinstance(state_index, tuple):
            state_index = self.state_indices[state_index]#
        state = self.state_lookup[state_index]

        actions = self.environment._possible_actions(cell=state)
        if len(actions) != len(self._get_neighbors(state_index)):
            raise ValueError

        # get Q values for actions
        q = np.array([self.Q[state_index, action] for action in actions])

        # select action maximizing the values
        actions_indices = np.where(q == q.max())[0] 
        return random.choice([actions[idx] for idx in actions_indices])