from loguru import logger
from enum import Enum, IntEnum
import matplotlib.pyplot as plt
import numpy as np

from rl.environment.render import Render
from rl.environment.actions import Actions, Action

class Cell(IntEnum):
    EMPTY = 0  # indicates empty cell where the agent can move to
    OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
    CURRENT = 2  # indicates current cell of the agent




class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class Environment():
    """ A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.

        The layout of the maze and the rules how to move through it are called the environment. An agent is placed
        at start_cell. The agent chooses actions (move left/right/up/down) in order to reach the shelter_cell. Every
        action results in a reward or penalty which are accumulated during the game. Every move gives a small
        penalty (-0.05), returning to a cell the agent visited earlier a bigger penalty (-0.25) and running into
        a wall a large penalty (-0.75). The reward (+10.0) is collected when the agent reaches the exit. The
        game always reaches a terminal state; the agent either wins or looses. Obviously reaching the exit means
        winning, but if the penalties the agent is collecting during play exceed a certain threshold the agent is
        assumed to wander around clueless and looses.

        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlib's plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """
    actions = Actions()  # all possible actions

    # plotting variales
    cell_marker_size = 150  # size of marker indicating shelter/start
    cell_scatter_size = 30  # size of scatter dots e.g. to show distances
    draw_ax = None  # ax to draw steps on
    __render = Render.NOTHING  # what to render
    _render_ax = None  # axes for rendering the moves

    # behaviour variables
    illegal_move = False  # check if last move was illegal
    stop_on_error = False  # check if play mode
    n_play_steps = 100  # max number of steps in play mode
    shelter_found = True  # keeps track of if the agent found the shelter
    at_shelter = False  # is the agent currently in the shelter cell?

    def __init__(self, maze, start_cell=(0, 0), shelter_cell=None):
        """ Create a new maze game.

            :param numpy.array maze: 2D array containing empty cells (= 0) and cells occupied with walls (= 1)
            :param tuple start_cell: starting cell for the agent in the maze (optional, else upper left)
            :param tuple shelter_cell: exit cell which the agent has to reach (optional, else lower right)
        """
        self.change_layout(maze, shelter_cell=shelter_cell)
        self.start_cell = start_cell
        self.reset(start_cell)

        self.__minimum_reward = -0.5 * self.maze.size

    @property
    def n_cells(self):
        return len(self.cells)

    @property
    def n_empty(self):
        return len(self.empty)

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def _cells(self):
        '''
            empty cells + end cells
        '''
        return self.empty + [self.shelter_cell]

    def _cell_empty_index(self, cell):
        return self.empty.index(cell)

    def _possible_actions(self, cell=None):
        """ Create a list with all possible actions from 'cell', avoiding the maze's edges and walls.

            :param tuple cell: location of the agent (optional, else use current cell)
            :return list: all possible actions
        """
        if cell is None:
            cell = np.array(self._current_cell)
        else:
            cell = np.array(cell)

        possible_actions = []
        for action in self.actions.actions:
            nxt = cell + action.shift
            if self.maze[nxt[1], nxt[0]] != Cell.OCCUPIED:
                possible_actions.append(action)

        return possible_actions

    def _impossible_actions(self, cell=None):
        '''
            Returns a list of actions that are illegal in a given state
        '''
        possible = self._possible_actions(cell=cell)
        impossible = [act for act in self.actions.actions.copy() if act not in possible]
        return impossible

    def reset(self, start_cell=None):
        """ Reset the maze to its initial state and place the agent at start_cell.

            :param tuple start_cell: here the agent starts its journey through the maze (optional, else upper left)
            :return: new state after reset
        """
        self.__total_reward = 0
        self.currently_at_shelter = False

        start_cell = start_cell or self.start_cell
        
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.shelter_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self._previous_cell = self._current_cell = start_cell
        self._visited_cells = set()  # a set() only stores unique values

        if self.__render in (Render.TRAINING, Render.MOVES):
            self._render_maze()

        return tuple(self._agent_location().ravel())

    def play(self, model, start_cell=None, ax=None):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param tuple start_cell: agents initial cell (optional, else upper left)
            :return Status: WIN, LOSE
        """
        self.stop_on_error = True
        self.draw_ax = ax

        # restart
        state = self.reset(start_cell=start_cell)
        model.on_play_start(state)

        play_reward = 0
        states = []
        for step in range(self.n_play_steps):
            action = model.predict(state_index=state)
            action = self.actions[action]

            state, reward, status = self.step(action)
            states.append(np.array(state))
            play_reward = max(reward, play_reward)

            d = np.linalg.norm(np.array(self._current_cell) - np.array(self.shelter_cell))
            if status in (Status.WIN, Status.LOSE) or d<3:
                self.stop_on_error = False

                if status == Status.WIN or d<3:
                    status = Status.WIN
                    play_reward = self.reward_exit
                break
        
        model.on_play_end()
        self.stop_on_error = False

        # get escape arm
        if status != Status.WIN or d>3:
            escape_arm = None
        else:
            trajectory = np.vstack(states)
            if np.max(trajectory[:, 0]) > 30:
                escape_arm = 'right'
            else:
                escape_arm = 'left'
        logger.debug(f'Finished play with escape arm: {escape_arm}')
        return status, step, play_reward, escape_arm, states

    def step(self, action, cell=None):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param Action action: the agent will move in this direction
            :return: state, reward, status
        """
        reward = self.execute_action(action, cell=cell)

        self.__total_reward += reward
        status = self._game_status()
        state = tuple(self._agent_location().ravel())
        return state, reward, status

    def execute_action(self, action, cell=None):
        """ Execute action and collect the reward or penalty.

            :param Action action: direction in which the agent will move
            :return float: reward or penalty which results from the action
        """
        if cell is not None:
            self._current_cell = cell

        if not isinstance(action, Action):
            action = self.actions[action]

        possible_actions = self._possible_actions(self._current_cell)
        if not possible_actions:
            raise ValueError('No possible actions')

        elif action in possible_actions:
            state = np.array(self._current_cell)
            next_state = state + action.shift

            self._previous_cell = self._current_cell
            self._current_cell = tuple(next_state)

            if self.__render != Render.NOTHING:
                self.__draw()

            # check if the agent detected the shelter
            d = np.linalg.norm(np.array(self._previous_cell) - np.array(self.shelter_cell))
            if d < 3:
                self.currently_at_shelter = True
            else:
                self.currently_at_shelter = False
            if not self.shelter_found and d < 3:
                    logger.debug('Agent found the shelter, good job.')
                    self.shelter_found = True
            
            if self._previous_cell == self.shelter_cell:
                self.at_shelter = True
            else:
                self.at_shelter = False

            # get reward
            # if self._current_cell == self.shelter_cell:
            if d < 3:
                reward = self.reward_exit  # maximum reward when reaching the exit cell
            else:
                if self._current_cell in self._visited_cells:
                    reward = self.penalty_visited  # penalty when returning to a cell which was visited earlier
                else:
                    reward = self.penalty_move  # penalty for a move which did not result in finding the exit cell

                # reward for geo/euclidean distance from shelter
                if self.shelter_found:
                    reward += self.geodesic[self._cell_empty_index(self._current_cell)] * self.reward_geodesic
                    reward += self.euclidean[self._cell_empty_index(self._current_cell)] * self.reward_euclidean

            self._visited_cells.add(self._current_cell)
            self.illegal_move = False
        else:
            # logger.warning(f'Attempted illegal move {action}: not in possible actions: {possible_actions}')
            self.illegal_move = True
            reward = self.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

        return reward

    def _game_status(self):
        """ Return the game status.

            :return Status: current game status (WIN, LOSE, PLAYING)
        """
        if self.illegal_move and self.stop_on_error:
            # can't make illegal moves
            logger.debug('In playing mode an illegal move was detected, stopping.')
            return Status.LOSE

        if self._current_cell == self.shelter_cell or self.currently_at_shelter:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            return Status.LOSE

        return Status.PLAYING

    def _agent_location(self):
        """ Return the state of the maze - in this game the agents current location.

            :return numpy.array [1][2]: agents current location
        """
        return np.array([[*self._current_cell]])

    def _render_maze(self, distance=None, ax=None):
        ''' render the maze '''
        nrows, ncols = self.maze.shape

        ax = ax or self._render_ax
        if ax is None:
            return

        # setup axes
        ax.set_xticks(np.arange(0.5, nrows, step=1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0.5, ncols, step=1))
        ax.set_yticklabels([])
        ax.grid(False)

        # mark start and end cells
        ax.scatter(*self.start_cell, s=self.cell_marker_size, zorder=20, color='green', edgecolors=[.2, .2, .2], lw=1.5)  # start is a big red square
        ax.text(*self.start_cell, "Start", ha="center", va="center", color="black", zorder=22)
        ax.scatter(*self.shelter_cell, s=self.cell_marker_size, zorder=20, color='red', edgecolors=[.2, .2, .2], lw=1.5)  # exit is a big green square
        ax.text(*self.shelter_cell, "Shelt", ha="center", va="center", color="black", zorder=22)

        # show maze
        ax.imshow(self.maze, cmap="binary")

        # mark distance
        if distance is not None:
            if distance == 'euclidean':
                dist = self.euclidean
            else:
                dist = self.geodesic
            
            x = [x for (x,y) in self.empty]
            y = [y for (x,y) in self.empty]
            ax.scatter(x, y, c=dist, cmap='bwr', s=self.cell_scatter_size, edgecolors=[.2, .2, .2], lw=1)
            ax.set(title=distance)

    def __draw(self):
        """ Draw a line from the agents previous cell to its current cell. """

        ax = self.draw_ax or self._render_ax
        ax.plot(*zip(*[self._previous_cell, self._current_cell]), "bo-", alpha=1, markersize=5, lw=3)  # previous cells are blue dots
        ax.plot(*self._current_cell, "ro", markersize=5)  # current cell is a red dot

    def render(self, content=Render.NOTHING, ax=None):
        """ Record what will be rendered during play and/or training.

            :param Render content: NOTHING, TRAINING, MOVES
        """
        self.__render = content
                
        if self.__render in (Render.MOVES, Render.TRAINING):
                self._render_ax = ax

        plt.show(block=False)
