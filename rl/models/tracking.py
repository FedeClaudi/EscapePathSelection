from pathlib import Path
import pandas as pd
from celluloid import Camera
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import random

from fcutils.progress import track
from myterial import teal


import sys
sys.path.append('./')
from rl.environment.actions import Actions
from rl.environment.render import Render
from rl.models.Q import QLearner

data_folder = Path('/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/analysis_metadata/explorations')

SCALING_FACTOR = 45 / 1000  # maze images are 50 wide and tracking is from images 1000 wide

def find_nearest(array, value):
    idx = np.linalg.norm(array - value, axis=1).argmin()
    return array[idx, :]

def tuplify(arr):
    ''' return an array as a tuple '''
    if isinstance(arr, tuple):
        return arr
    return tuple(arr.flatten())

class QLearnerTracking(QLearner):
    '''
        Base class for Q learning agents that learn by following 
        the steps of mice.
    '''
    
    use_diagonal = True  # otherwise decomposes them into two normal steps
    def __init__(self, environment, maze_name, trial_number=None, take_all_actions=False, **kwargs):
        '''
            Class to help RL on real tracking data. It loads and process the data to 
            prepare for RL on them and it has a dedicated training loop specific to tracking data.

            Arguments:
                environment: Maze.
                maze_name: str. Name of the experimental maze
                trial_numnber: int. Trial/session number whos data are to be used
                take_all_actions: bool. If true at each step the agent tries to take all actions.
                    Mimicking mice looking around and sniffing around.
        '''
        super().__init__(environment, **kwargs)

        self.tracking_errors = []
        self.take_all_actions = take_all_actions

        # tell the environment to wait for rewards until the agent found the shelter
        self.environment.shelter_found = False

        self.params['maze_name'] = maze_name
        self.params['trial_number'] = trial_number
        self.params['take_all_actions'] = take_all_actions
        self.tracking = self.get_tracking(maze_name, trial_number=trial_number)

    @property
    def n_trials(self):
        return len(self.tracking_data)

    # ----------------------------- Get tracking data ---------------------------- #

    def get_tracking(self, maze_name, trial_number=None):
        '''
            Get tracking data from a given trial in a given maze.
            If trial number is not given a random trial is chosen

            Tracking data are 'quantized' by getting the closest cell in the maze at a given time point
        '''
        # get tracking data
        data = pd.read_hdf(data_folder/ f'{maze_name}_explorations.h5', key='hdf')
        self.tracking_data = data
        trial = data.iloc[trial_number] if trial_number is not None else data.sample(1)

        # scale down tracking data
        scaling_factor = self.environment.tracking_scaling_factor or SCALING_FACTOR
        trial_tracking = trial.body_tracking[:, :2] * scaling_factor
        N = trial.body_tracking.shape[0]
        logger.debug(f'Loading tracking data from {maze_name} trial {trial_number}. {N} samples')

        # remove nans
        x = pd.Series(trial_tracking[:, 0]).interpolate().values
        y = pd.Series(trial_tracking[:, 1]).interpolate().values
        trial_tracking = np.vstack([x, y]).T

        # flip tracking on the y axis
        y = - (trial_tracking[:, 1] - 25) + 23
        trial_tracking[:, 1] = y - 3 + self.environment.tracking_y_shift
        trial_tracking[:, 0] += 2 + self.environment.tracking_x_shift
        
        # organize maze cells
        cells = np.vstack(self.environment.empty)

        # get closest cell at each time point
        prev = 0
        tracking = []
        for frame in range(N):
            # get closest cell
            cell = find_nearest(cells, trial_tracking[frame])

            if np.linalg.norm(cell - prev) > 10 and np.all(prev != 0):
                self.tracking_errors.append(frame)

            if np.any(cell != prev): # only do stuff is state changed
                if self.use_diagonal:
                    tracking.append(cell)
                    prev = cell
                else:
                    # check if a diagonal movement was done
                    if frame > 0:
                        dx = cell[0] - prev[0]
                        dy = cell[1] - prev[1]

                        if abs(dx) == 1 and abs(dy) == 1:
                            steps = self._decompose_diagnonal_step(prev, cell, dx, dy)
                            tracking.extend(steps)
                            prev = steps[-1]
                        else:
                            tracking.append(cell)
                            prev = cell
                    else:
                        tracking.append(cell)
                        prev = cell
        tracking = np.vstack(tracking)

        self.trial_tracking = trial_tracking
        return tracking
    
    def _check_step_ok(self, cell, dx=0, dy=0):
        '''
            Checks if a step in dx or dy leads to a cell that is in the maze
        '''
        cell = cell.copy() + np.array([dx, dy])
        if tuplify(cell) in self.environment.empty:
            return cell
        else:
            return None

    def _decompose_diagnonal_step(self, prev, cell, dx, dy):
        '''
            Decompose a diagonal step into a vertical and a horizontal step
            taking care of not stepping outside maze
        '''
        # try to move in horizontal direction first
        first = self._check_step_ok(prev, dx=dx, dy=0)

        if first is None:
            # that was a bad idea, move in vertical direction first
            first = self._check_step_ok(prev, dx=0, dy=dy)
            if first is None:
                raise ValueError('Could not move in either direction!')

            second = self._check_step_ok(first, dx=dx, dy=0)
        else:
            second = self._check_step_ok(first, dx=0, dy=dy)

        if second is None:
            raise ValueError('Second step went wrong')

        if np.any(second != cell):
            raise ValueError(f'Second step {second} is different from final cell {cell}')
        return [first, second]

    def plot_tracking(self, ax=None, plot_raw=True):
        '''
            Plot maze, tracking and quantized tracking overimposed to check quality
        '''
        if ax is None:
            f, ax = plt.subplots(figsize=(9, 9))
            ax.set(xticks=[], yticks=[])

        if plot_raw:
            ax.plot(self.trial_tracking[:, 0], self.trial_tracking[:, 1], lw=2, color=teal, alpha=.4, label='tracking', zorder=110)

        ax.plot(self.tracking[:, 0], self.tracking[:, 1], lw=2, color='salmon', alpha=1, label='quantized tracking')
        self.environment._render_maze(ax=ax)
        ax.legend()


    # ------------------------------- training loop ------------------------------ #
    def train(self, random_start=True, film=False, **kwargs):
        '''
            Main agent training loop. Iterates over N episodes of max M steps.
            At each step the agent takes an action and updates the Q table

            Arguments:
                random_start: bool. If true a random start cell is used at each episode
                film: bool. If true a video is created showing the trajectory at each step.
        '''
        _stop = self.environment.stop_on_error
        self.environment.stop_on_error = self.stop_on_error

        # make camera
        if film:
            f, axes = plt.subplots(figsize=(12, 8), ncols=2, tight_layout=True)
            camera = Camera(f)
            self.environment.render(Render.TRAINING, ax = axes[0])

        # training starts here
        start_list = list()
        logger.debug(f'Starting training by following mouse for {len(self.tracking)} steps')
        state = self.environment.reset(tuplify(self.tracking[0, :]))
        state_index = self.state_indices[state]
        _cell = None  # used to reset environment stuff when a tracking jump is detected
        n_steps = self.tracking.shape[0]-1

        for step in track(range(n_steps), total=n_steps, description=f'Training {self.name}', transient=True):
            # if step > 320:
            #     logger.warning('!!!!! DEBUG BREAKKKDFIUASHDUIASBERaswfhgiubfgiuewhfiu')
            #     break

            # check if there's a jump in the tracking
            if self.check_jump(step):
                # just move the agent to the next state and update the environment
                state = _cell = tuplify(self.tracking[step+1, :])
                self.environment._current_cell = _cell
                continue  # skip this frame

            # get the action that matches what the mouse did
            action = self.get_action(step)
            if action is None:
                # diagnonal step?
                logger.debug('No action, skipping')
                continue
            
            # execute fake actions
            if self.take_all_actions:
                _cell = state
                # have the agent take all actions and learn from them, but don't move the agent
                for fake_action in self.environment.actions:
                    if fake_action != action:
                        _next_state, reward, _ = self.environment.step(fake_action, cell=state)
                        if _next_state != state:
                            # update agent
                            next_state_index = self.state_indices[_next_state]
                            self.tracking_training_step(state_index, fake_action, next_state_index, reward)
            
            # execute action and check everything's ok
            next_state, reward, status = self.environment.step(action, cell=_cell)

            if next_state == state:
                raise ValueError(f'At step {step} took action {self.actions[action]} but the state didnt change: {state}')

            _cell = None
            if np.any(np.array(next_state) != self.tracking[step+1]):
                raise ValueError(f'Next state is different from next tracking point: {next_state} instead of {self.tracking[step+1]}')

            next_state = tuplify(next_state)
            if next_state == state:
                raise ValueError('Next state is same as current state, what the heck')

            # update agent
            next_state_index = self.state_indices[next_state]
            self.tracking_training_step(state_index, action, next_state_index, reward)

            # update state
            state = next_state
            state_index = self.state_indices[state]

            # snap
            if film:
                if step % 5 == 0 or step == self.tracking.shape[0]:
                    axes[0].imshow(self.environment.maze, cmap="binary")
                    axes[1].text(-.5, -.7, f'{step} training frame - action: {actions[action]}', fontsize=15, color='k')
                    self.environment.render_q(self, ax=axes[1], showmaze=True, global_scale=False)
                    camera.snap()

        if film:
            axes[0].get_figure().canvas.draw()
            axes[0].get_figure().canvas.flush_events()

            animation = camera.animate()
            logger.debug('Saving training.mp4')
            animation.save('training.mp4')

        self.environment.stop_on_error = _stop
        
        # conclusion
        self.on_training_end()

    def tracking_training_step(self, state_index, action, next_state_index, reward):
        raise ValueError("An agent being trained on tracking data should implement:\n"
                            "def tracking_training_step(self, state_index, action, next_state_index, reward):")

    def check_jump(self, step):
        '''
            Checks if there's a jump in the tracking
        '''
        p0 = self.tracking[step, :]
        p1 = self.tracking[step+1, :]

        if abs(p0[0] - p1[0]) > 1 or abs(p0[1] - p1[1]) > 1:
            logger.debug(f'Jump at step {step}, moved from {p0} to {p1}')
            return True
        else:
            return False

    def get_action(self, step):
        '''
            During training from tracking data, get which action
            was performed based on where we go next.
        '''
        p0 = self.tracking[step, :]
        p1 = self.tracking[step+1, :]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        return Actions.given_delta((dx, dy))
