import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
from loguru import logger
import random

from fcutils.progress import track

from environment import Status
from rl.models.Q import TrainingHistory


class SmdpQlearner(object):
    lr = .2

    def __init__(self,
            environment,
            states_indices,
            n_actions=5,
            n_options=4,
            options=None,
            max_n_steps = 1000,
            episodes = 100,
            discount = 0.9999,
            exploration_rate = .4,
        ):
        self.name = 'SuccessorOptions'
        
        # setup sMDP params
        self.options = options
        self.options_Qs = [o.Q for o in self.options]
        self.states_indices = states_indices

        self.environment = environment
        self.n_actions = n_actions
        self.n_options = n_options
        self.n_states = self.environment.n_cells

        # create Q table
        self.Q = np.zeros((self.n_states, self.n_actions + self.n_options))

        # training parameters
        self.max_n_steps = max_n_steps
        self.episodes = episodes
        self.discount = discount
        self.exploration_rate = exploration_rate

        # reset environment
        self.environment.reset()

    def done(self, status):
        return status in (Status.WIN, Status.LOSE)

    def _train_primitive_action(self, action, state):
        '''
            take a primitive action and learn from it.
            action: int with action ID
            state: int with state ID
        '''
        # take primitive action
        next_state, reward, status = self.environment.step(action)
        next_state = self.states_indices[next_state]

        # update training history
        self.training_history.on_step_end(reward)
        
        # get next action
        next_action = self.get_greedy_action(next_state)

        # update Q table
        self.Q[state, action] = self.lr * self.Q[state, action] + \
            (1 - self.lr) * (reward + self.discount * self.Q[next_state, next_action])
        return next_state, status

    def _train_option(self, action, state):
        '''
            Execute actions according to the option selected until we reach a state
            in the option's termination set. Keep track of the comulative reward
            accumulated while following the the option and use that to update 
            the Q value at the original state
        '''
        start_state = state
        option = self.options[action - self.n_actions -1]
        status = Status.PLAYING

        # do some halth checks on the option
        if not option.subgoal_state_index in option.termination_set:
            raise ValueError(f'The goal state is not in the termination set for: {option}')
        
        # play the option until the termination set is reached
        option_reward = 0
        option_steps = 0
        while state in option.initiation_set:
            option_steps += 1
            
            # select action
            action = option.get_action(state)

            # execute action
            next_state, reward, status = self.environment.step(action)
            next_state = self.states_indices[next_state]
            option_reward += reward

            # update training history
            self.training_history.on_step_end(reward)

            # check if done
            if (
                self.done(status) or 
                self.training_history._episode_steps > self.max_n_steps-1 or
                next_state == state
            ):  
                return option_steps, state, status, start_state, option_reward
            else:
                # update state and repeat
                state = next_state
        return option_steps, state, status, start_state, option_reward


    def train(self, episodes=None, **kwargs):
        """
        Train using SMDP-Q-learning

        The following update are used
            * Execute action a 
                - Update action a using normal Q-learning update
                - For all options, check if opt(o, s) = a
                - If true:
                    Q(s, o)  = a*Q(s,o) + (1-a)*(r + Q(s', o)) if s' non-terminal
                    Q(s, o)  = a*Q(s,o) + (1-a)*(r + max_a Q(s', a)) if s' terminal

            * Execute option o:
                - Q(s, o)  = a*Q(s,o) + (1-a)*(r + Q(s', o)) if s' non-terminal
                - Q(s, o)  = a*Q(s,o) + (1-a)*(r + max_a Q(s', a)) if s' terminal
                - Q(s, a) for all (s,a) pairs encountered while executing option
                - If opt(o', s) = a = opt(o, s)
                    Q(s, o')  = a*Q(s,o') + (1-a)*(r + Q(s', o') if s' non-terminal
                    Q(s, o')  = a*Q(s,o') + (1-a)*(r + max_a Q(s', a)) if s' terminal
        """        
        self.training_history = TrainingHistory()

        # get number of episodes
        episodes = episodes or self.episodes
        logger.debug(f'SmpdQLearner training for {episodes} episodes')

        for episode in track(range(episodes), total=episodes, description=f'Training {self.name}', transient=True):
            # reset things
            state = self.states_indices[self.environment.reset()]
            steps_count = 0
            self.training_history.on_episode_start()

            while True:
                # select action or option with epsilong reedy
                if np.random.random() < self.exploration_rate:
                    #  Uniformly random
                    action = np.random.randint(0, self.n_options + self.n_actions)
                else:
                    #  Greedy
                    action = self.get_greedy_action(state)

                # either perform a primitive or several according to an option
                if action < self.n_actions:
                    # execute primitive action and learn from it
                    state, status = self._train_primitive_action(action, state)
                    steps_count += 1
                else:
                    # Execute the option and learn from it
                    k, state, status, option_start, option_reward = self._train_option(action, state)
                    steps_count += k

                    # update Q value of taking option
                    max_next_Q = self.Q[state].max()
                    self.Q[option_start, action] += self.lr * (
                                option_reward + self.discount * max_next_Q - self.Q[option_start, action]
                    )

                if self.done(status) or steps_count >= self.max_n_steps-1:
                    state = self.environment.reset()
                    next_state = self.states_indices[state]

                    # restart training history
                    self.training_history.on_episode_end(status)
                    self.training_history.on_episode_start()
                    break

        return self.Q, self.training_history

    def get_greedy_action(self, state):
        '''
            Returns the action/option with highest
            Q value.
        '''
        qvals = self.Q[state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def get_option_greedy_action(self, state, op):
        '''
            Returns the action with highest value 
            according to an option's policy
        '''
        op = op - self.n_actions
        qvals = self.options_Qs[op][state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def get_option_actions(self, state, op):
        '''
            Returns a list of option actions with highest values
        '''
        op = op - self.n_actions
        qvals = self.options_Qs[op][state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        return list(maxvals)


