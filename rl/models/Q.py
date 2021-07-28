import numpy as np
import random
from celluloid import Camera
import matplotlib.pyplot as plt
from loguru import logger
from rich.table import Table

from myterial import orange, pink_light, blue_light, green, salmon
from myterial.utils import map_color
from fcutils.progress import track

from rl.environment.render import Render
from rl.environment.environment import Status
from rl.environment.actions import Actions
from rl.models.trainig_history import TrainingHistory


class QLearner:
    '''
        Base class for all agents based on Q learning
    '''
    stop_on_error = False  # if true during training stop when an incorrect action is performed\
    deterministic = False  # if true during training no random actions are taken

    def __init__(self,
                maze,
                name='',
                max_n_steps = 100,
                discount = 0.9999,
                exploration_rate = 0.3,
                learning_rate = .3,
                episodes = 100,
                **kwargs        
        ):
        self.training_mode = False
        self.environment = maze
        self.name = name

        # initialize Q
        self.Q = np.zeros((self.environment.n_cells, self.environment.n_actions))

        # max (x,y) states to indices and vice versa
        self.state_indices = {state:n for n, state in enumerate(self.environment.cells)}
        self.state_lookup = {v:k for k,v in self.state_indices.items()}

        # set parameters
        self.max_n_steps = max_n_steps
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.episodes = episodes

        self.params = dict(
            max_n_steps = self.max_n_steps,
            discount = self.discount,
            exploration_rate = self.exploration_rate,
            learning_rate = self.learning_rate,
            episodes = self.episodes,
        )


    def __rich_console__(self, console, measure):
        tb = Table(show_lines=False, box=None)
        tb.add_column('Parameter', justify='right', header_style='bold yellow', style=pink_light)
        tb.add_column('Value', justify='left', header_style='bold yellow', style=blue_light)
        for param, value in self.params.items():
            tb.add_row(param, str(value))

        yield f'Model: [b {orange}]{self.name}'
        yield tb
        yield '\n\n'
       
    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if isinstance(state, tuple):
            state = self.state_indices[state]

        return self.Q[state, :]

    @property
    def q_max(self):
        ''' max value of any state action pair '''
        return self.Q.max()

    @property
    def q_min(self):
        ''' min value of any state action pair '''
        return self.Q.min()

    def state_q_max(self, state_index):
        '''
            Max Q value at a state
        '''
        return self.Q[state_index].max()

    def on_reset(self, state):
        '''
            Called when self.reset() is called
        '''
        return

    def on_training_end(self):
        '''
            Called when training is completed, to be replace by models
        '''
        return

    def on_play_start(self, state):
        '''
            Called before playing is started
        '''
        return

    def on_play_end(self):
        '''
            Called when playing is completed, to be replace by models
        '''
        return

    def predict(self, state_index=0):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state_index: game state_index
            :return int: selected action
        """
        q = self.q(state_index)
        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)


    # ------------------------------- Training loop ------------------------------ #
    def train(self, random_start=True, film=False, episodes=None, test_performance=False):
        '''
            Main agent training loop. Iterates over N episodes of max M steps.
            At each step the agent takes an action and updates the Q table

            Arguments:
                random_start: bool. If true a random start cell is used at each episode
                film: bool. If true a video is created showing the trajectory at each step.
                episodes: int. Number of episodes to train for.
        '''
        self.training_mode = True
        self.training_history = TrainingHistory()

        _stop = self.environment.stop_on_error
        self.environment.stop_on_error = self.stop_on_error

        # make camera
        if film:
            f, axes = plt.subplots(figsize=(12, 8), ncols=2, tight_layout=True)
            camera = Camera(f)
            self.environment.render(Render.TRAINING, ax = axes[0])

        # training starts here
        start_list = list()
        episodes = episodes or self.episodes
        
        # for episode in track(range(episodes), total=episodes, description=f'Training {self.name}', transient=True):
        for episode in range(episodes):
            self.training_history.on_episode_start()

            # select start cell
            if random_start:
                # ? optimization: make sure to start from all possible cells
                if not start_list:
                    start_list = self.environment.empty.copy()
                start_cell = random.choice(start_list)
                start_list.remove(start_cell)
            else:
                start_cell = self.environment.start_cell

            # reset environment and agent
            state_index = self.state_indices[self.environment.reset(start_cell)]
            self.on_reset(self.state_lookup[state_index])

            # iterate over steps until done
            for step in range(self.max_n_steps):
                # agent step
                next_state_index, reward, status = self.training_step(state_index)

                # keep track of training progress
                self.training_history.on_step_end(reward, self.state_lookup[state_index])

                # check if DONE
                if status in (Status.WIN, Status.LOSE) or step >= self.max_n_steps-1:  # terminal state reached, stop training episode
                    if step == 0 and not random_start: raise ValueError(f'Ended after 0 steps, how? Started at: {start_cell}')
                    logger.debug(f'Finishing episode {episode} with status: {status} after {step} steps')
                    break
                
                # update state
                state_index = next_state_index

            self.exploration_rate *= .995

            # test performance in play mode
            if test_performance:
                status, step, reward, arm, states = self.environment.play(self, start_cell=self.environment.start_cell)
                self.training_history.update_play_history(status, step, reward, arm)

            # update training history
            self.training_history.on_episode_end(status)

            if film:
                axes[0].text(-.5, -.7, f'{episode+1} training iterations', fontsize=20, color='k')
                self.environment.render_q(self, ax=axes[1])
                camera.snap()
            self.training_mode = False

        # save video
        if film:
            animation = camera.animate()
            logger.info('Saving training.mp4')
            animation.save('training.mp4')

        self.environment.stop_on_error = _stop

        # conclusion
        self.on_training_end()

    def training_step(self, state_index):
        '''
            Single step during training of one episode.

            Arguments:
                state_index: int. State index
        '''
        raise ValueError('Need to implement training step function!')

    # ------------------------- training helper functions ------------------------ #
    def choose_epsilon_greedy_action(self, state_index):
        '''
            Choose an action at a state with an epsilon greedy
            strategy.
        '''
        # choose action epsilon greedy (off-policy, instead of only using the learned policy)
        if np.random.random() < self.exploration_rate and not self.deterministic:
            action = random.choice(self.environment.actions)
        else:
            action = self.predict(state_index)

        return action


    # --------------------------------- plotting --------------------------------- #
    def render_q(self, ax=None, showmaze=True, global_scale=False, hw=0.5, hl=0.1, max_only=False, cmap='bwr'):
        """ Render the recommended action(s) for each cell as provided by 'model'.
        """
        
        # set axis appearance
        ax = ax or self.environment.__ax2
        nrows, ncols = self.environment.maze.shape
        ax.set_xticks(np.arange(0.5, nrows, step=1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0.5, ncols, step=1))
        ax.set_yticklabels([])
        ax.grid(False)

        actions = Actions()
        for cell in self.environment.empty:
            q = self.q(cell)

            if len(q) == 0:
                continue

            a = np.nonzero(q == np.max(q))[0]

            qmin, qmax = np.min(q), np.max(q)

            for action in np.arange(len(q)):
                if q[action] == 0: continue
                if max_only and q[action] != qmax:
                    continue
                dx, dy = actions[action].shift / 3
                

                # get color
                # red: high certainty positive, blue high certainty negative
                if global_scale:
                    c = map_color(q[action], name=cmap, vmin=qmin, vmax=qmax)
                else:
                    if q[action] == qmax:
                        c = green
                    else:
                        c = map_color(q[action], name=cmap, vmin=qmin, vmax=qmax)

                ax.arrow(*cell, dx, dy, color=c, head_width=0.4, head_length=0.1)

        if showmaze:
            ax.imshow(self.environment.maze, cmap="binary")
            ax.scatter(*self.environment.shelter_cell, s=self.environment.cell_marker_size, zorder=20, lw=1.5, color='green', edgecolors=[.2, .2, .2])  # exit is a big green square
            ax.text(*self.environment.shelter_cell, "Shelt.", ha="center", va="center", color="black", zorder=22)
            ax.scatter(*self.environment.start_cell, s=self.environment.cell_marker_size, zorder=20, lw=1.5, edgecolors=[.2, .2, .2], color='red', alpha=.15)  # start is a big red square
