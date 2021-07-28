import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import random
from loguru import logger 

from fcutils.progress import track
from myterial.utils import map_color
from myterial import green, indigo, orange

from rl.environment.environment import Status

class SimpleQLearner(object):
    def __init__(self, environment, reward_vector, state_indices, option, n_actions=4):
        """
        Q-learner that trains on primitive actions
        Use to train the option policies
        """
        self.environment = environment
        self.reward_vector = reward_vector
        self.state_indices = state_indices
        self.Q = np.zeros((len(reward_vector), n_actions))
        self.option = option

    def train(self, episodes=100, steps=1000, random_start=True):
        """
        Run iterations to obtain optimal policy using Q-learning
        """

        exploration_rate = .4
        learning_rate = .01
        discount = .9999
        start_list = list()

        for episode in track(range(episodes), total=episodes, description='simple qlearning', transient=True):
            # selet start state
            if random_start:
                # ? optimization: make sure to start from all possible cells
                if not start_list:
                    start_list = self.environment.empty.copy()
                start_cell = random.choice(start_list)
                start_list.remove(start_cell)
            else:
                start_cell = self.environment.start_cell
            state = self.environment.reset(start_cell)
            
            for step in range(steps):
                # select action
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.get_greedy_action(state)

                # execute action
                next_state, _, status = self.environment.step(action)
                
                # update Q
                state_idx = self.state_indices[state]
                reward = self.reward(state_idx, self.state_indices[next_state])

                max_next_Q = self.Q[state, :].max()
                self.Q[state_idx, action] += learning_rate * \
                            (reward + discount * max_next_Q - self.Q[state_idx, action])

                # check if done
                if next_state == self.option.subgoal_state:
                    break
                else:
                    state = next_state
        return self.Q

    def reward(self, s1, s2):
        """
        The reward for the given SR
        """
        r1 = self.reward_vector[s1]
        r2 = self.reward_vector[s2]
        return (r2 - r1)

    def get_greedy_action(self, state):
        """
        Get action that maximizes Q-value
        """
        qvals = self.Q[state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def save(self, path):
        """
        Save the Q-values of the policy into pickle file
        """
        logger.debug(f'Saving policy at: {path}')
        with open(path, "wb") as fp:
            pickle.dump((self.Q, self.state_indices), fp)

    def plot_reward_function(self):
        f, ax = plt.subplots()
        r = self.reward_vector.copy()
        r[r==0] = np.nan
        ax.imshow(r.reshape(50, 50).T)