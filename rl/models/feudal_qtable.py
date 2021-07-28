import numpy as np
import random
import pandas as pd

from myterial.utils import map_color
from environment import Status
from rl.environment.render import Render
from rl.environment.actions import Actions

from rl.models.Q import QLearner

class FeudalQLearningTable(QLearner):
    def __init__(self, game, name='Feudal', *args, n_layers=4, **kwargs):
        super().__init__(game, name='Feudal', **kwargs)

        # parameters
        self.number_actions = self.environment.n_actions
        self.number_layers = n_layers

        # initialize feudal layers
        self.levels = {0: FeudalLevel(actions=[self.number_actions], **kwargs)}
        if(n_layers > 1):
            for i in range(1, n_layers-1):
                self.levels[i] = FeudalLevel(
                    actions=list(range(self.number_actions+1)), **kwargs)
            self.levels[n_layers -
                        1] = FeudalLevel(actions=list(range(self.number_actions)), **kwargs)


    def choose_action(self, state):
        '''
            For each subordinate level, select an action
        '''
        level_states = self.get_level_states(state)
        actions = []
        for a in range(self.number_layers):
            actions.append(self.levels[a].choose_action(level_states[a]))
        return actions


    def training_step(self, state_index):
        '''
            Single step during training of one episode.

            Arguments:
                state_index: int. State index
        '''
        state = self.state_lookup[state_index]

        # select an action
        actions = self.choose_action(state)

        next_state, reward, status = self.environment.step(self.environment.actions[actions[-1]])
        next_state_index = self.state_indices[next_state]

        # learn
        done = status == Status.WIN
        self.learn(state, actions, reward, next_state, done)

        return next_state_index, reward, status


    def learn(self, s, actions, r, s_, done):
        level_states = self.get_level_states(s)
        level_states_prime = self.get_level_states(s_)

        # rewards for subordinates obeying their managers
        obey_reward = 0
        not_obey_reward = -1

        # teach them layerz
        for i in range(self.number_layers):
            if i == 0:
                reward = r
            else:
                if actions[i-1] == 4:
                    reward = r
                else:
                    if actions[i-1] == actions[i]:
                        reward = obey_reward
                    else:
                        reward = not_obey_reward

            self.levels[i].learn(level_states[i], actions[i],
                                 reward, level_states_prime[i], done)

    def get_level_states(self, state):
        states = []
        states.append(state)
        for i in range(self.number_layers-2, -1, -1):
            states.append((int(states[-1][0]/2), int(states[-1][1]/2)))
        states.reverse()
        return states

    @property
    def top_Q(self):
        ''' Get the Q table for the level at highest level of resolution '''
        # get Q table as a pandas data frame
        Q = list(self.levels.values())[-1].q_table
        return Q


    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        try:
            values = self.top_Q.loc[str(state)].values
        except KeyError:
            # didn't visit this state during training?
            logger.debug(f'Didnt find Q values for state {state}')
            values = np.array([0, 0, 0, 0])

        return values

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: game state
            :return int: selected action
        """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())
            
        q = self.top_Q.loc[str(state)].values
        actions = np.nonzero(q == np.max(q))[0]
        return random.choice(actions)

    def plot_all_Q_tables(self):
        f, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 9))

        vmin, vmax = self.q_min, self.q_max
        for l, (level, ax) in enumerate(zip(self.levels.values(), axes.flatten())):
            ax.invert_yaxis()

            for state, actions in level.q_table.iterrows():
                for n, action in enumerate(actions.values):
                    if n > 3:
                        continue  # ignore special action
                    # n = n - 1
                    state = state.replace('(','').replace(')', '')
                    x, y = state.split(', ')
                    dx = 0
                    dy = 0
                    if n == Action.MOVE_LEFT:
                        dx = -0.3
                    if n == Action.MOVE_RIGHT:
                        dx = +0.3
                    if n == Action.MOVE_UP:
                        dy = -0.3
                    if n == Action.MOVE_DOWN:
                        dy = 0.3

                    c = map_color(action, name='bwr', vmin=actions.min(), vmax=actions.max())
                    ax.arrow(int(x), int(y), dx, dy, color=c, head_width=0.4, head_length=0.1)
            ax.set(title=f'Feudal level {l}', xticks=[], yticks=[])



class FeudalLevel:
    def __init__(self, actions, **kwargs):
        self.actions = actions  # a list
        self.lr = kwargs.pop('learning_rate', 0.01)
        self.gamma = kwargs.pop('discount', 0.9)
        self.epsilon = kwargs.pop('exploration_rate', .3)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float_)

    def choose_action(self, observation):
        observation = str(observation)
        self.check_state_exist(observation)

        # action selection
        if np.random.random() > self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not done:
            # next state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )