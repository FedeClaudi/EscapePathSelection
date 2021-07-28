import numpy as np
from loguru import logger
import random

from myterial import orange

from environment import Status
from rl.environment.render import Render
from rl.environment.actions import Actions

from rl.models.successor_options.successor import Successor
from rl.models.successor_options.smdpqlearner import SmdpQlearner

PLOT = True  # show plots during learning ?

class SuccessorOptions(Successor):
    max_n_steps = 1000
    following_option = False  # during prediction, true if we are using an option
    option_steps = 0  # keep track of number of steps with an option
    previous_option_state = None

    def __init__(self, game, name='', n_clusters=4, **kwargs):
        self.environment = game
        self.name = name
        Successor.__init__(self)

        # create successor representation using random policy
        logger.debug('Random policy exploration to build SR')
        self.get_successor(iters=int(4e4), load=True)

        # cluster SR to find options
        logger.debug('Clustering SR')
        self.get_subgoals(n_clusters)

        # build a policy for each state
        logger.debug('Building policy')
        self.learn_option_policies(episodes=400, steps=500, load=True)

        # now train a sMDP q learning agent
        self.smdp_learner = SmdpQlearner(
                            self.environment, 
                            self.state_indices,
                            options=self.options, 
                            n_options=n_clusters,
                            n_actions = 4,  # there's the no-option action as well
                            max_n_steps = kwargs.pop('max_n_steps', 1000),
                            episodes=kwargs.pop('episodes', 100),
                            discount = kwargs.pop('discount', .9999),
                            exploration_rate = kwargs.pop('exploration_rate', .4),
        )

        if PLOT:
            self.plot_subgoals()
            self.plot_options()

    def __repr__(self):
        return 'SuccessorOptions'

    def __str__(self):
        return 'SuccessorOptions'

    def __rich_console__(self, *args):
        return f'[{orange}]SuccessorOptions'

    def train(self, **kwargs):
        self.Q, self.training_history = self.smdp_learner.train(**kwargs)

    def q(self, state):
        return self.Q[self.state_indices[state], :self.environment.n_actions]

    @property
    def qmax(self):
        return self.Q.max()

    @property
    def qmin(self):
        return self.Q.min()

    def predict_with_option(self, state):
        '''
            Get the primitive action suggested by the
            current option
        '''
        self.option_steps += 1
        Q = self.current_option.Q
        q = Q[self.state_indices[state], :]

        if q.max() <= 0:
            # stop following this option
            self.following_option = False
            logger.debug(f'Stopped following option {self.current_option} after {self.option_steps} steps')

        self.previous_option_state = state
        return random.choice(np.nonzero(q == np.max(q))[0])

    def predict(self, state):
        if not self.following_option:
            # get the option with highest Q value
            q = self.Q[self.state_indices[state], :]
            action = random.choice(np.nonzero(q == np.max(q))[0])

            # primitive action or policy?
            if action < self.environment.n_actions:
                # use primitive actions
                logger.debug(f'Predict: taking primitive action: {actions[action]}')
                return action
            else:
                # start option
                self.following_option = True
                option_index = action - self.environment.n_actions
                
                self.option_steps = 0

                if option_index < 0 or option_index > (len(self.options)+1):
                    raise ValueError(f'Incorrect option index: {option_index}')

                self.current_option = self.options[option_index]
                logger.debug(f'Started following option {self.current_option}')
                return self.predict_with_option(state)
        else:
            # predict with option or stop option
            return self.predict_with_option(state)
        return random.choice(actions)