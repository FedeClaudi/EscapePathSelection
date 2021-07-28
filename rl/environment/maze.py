from dataclasses import dataclass
from PIL import Image
from loguru import logger
try:
    import skfmm
except:
    logger.warning('Could not import skfmm')
import numpy as np
import networkx as nx

from rl.environment.environment import Environment, Cell
from rl.environment.render import Render

@dataclass
class MazeLayout():
    maze: np.ndarray  # maze layout
    MIN_N_STEPS: int  # min num steps to reach goal
    START: tuple  # coordinates of start cell
    SHELTER: tuple  # coordinates of goal cell
    description: str  # description of maze
    name: str  # maze name


class Maze(MazeLayout, Environment):
    tracking_scaling_factor = None  # for tracking data
    tracking_x_shift = 0
    tracking_y_shift = 0
    
    def __init__(self, maze, REWARDS, *args, **kwargs):
        '''
            Base class for mazes.

            Arguments:
                REWARDS: dict of rewards values
        '''
        MazeLayout.__init__(self, maze, *args, **kwargs)
        Environment.__init__(self, self.maze, start_cell=self.START, shelter_cell=self.SHELTER)

        self.reward_exit = REWARDS['reward_exit']
        self.penalty_move = REWARDS['penalty_move']
        self.penalty_visited = REWARDS['penalty_visited']
        self.penalty_impossible_move = REWARDS['penalty_impossible_move']
        self.reward_euclidean = REWARDS['reward_euclidean']
        self.reward_geodesic = REWARDS['reward_geodesic']

        self.render(Render.NOTHING)

    def _compute_geodesic(self):
        '''
            Computes the geodesic distance from the shelter at each point
        '''
        dist = geodist(self.maze, self.shelter_cell)
        self.geodesic = np.array([dist[y, x] for (x,y) in self.empty])
        self.geodesic = 1 - self.geodesic/np.nanmax(self.geodesic)

    def _compute_euclidean(self):
        '''
            Computes the euclidean distance from the shelter at each point
        '''
        g = np.array(self.shelter_cell)
        self.euclidean = np.array([np.linalg.norm(np.array([x, y]) - g) for (x,y) in self.empty])
        self.euclidean = 1 - self.euclidean/np.nanmax(self.euclidean)

    def change_layout(self, new_layout, shelter_cell=None):
        self.maze = new_layout

        self.__minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        
        if shelter_cell is not None:
            self.shelter_cell = (ncols - 1, nrows - 1) if shelter_cell is None else shelter_cell

            try:
                self.empty.remove(self.shelter_cell)
            except ValueError:
                raise ValueError(f'The exit cell {shelter_cell} is not in the list of empty maze cells')

            # Check for impossible maze layout
            if self.shelter_cell not in self.cells:
                raise Exception("Error: exit cell at {} is not inside maze".format(self.shelter_cell))
            if self.maze[self.shelter_cell[::-1]] == Cell.OCCUPIED:
                raise Exception("Error: exit cell at {} is not free".format(self.shelter_cell))

        # compute geodesic and euclidean distance at each point
        self._compute_geodesic()
        self._compute_euclidean()

    def build_graph(self):
        '''
            Builds a graph representation of the maze
        '''
        # create graph and add nodes
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self._cells)

        # add edges
        delta = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for state in self._cells:
            for shift in delta:
                next_state = tuple(np.array(state) + np.array(shift))
                if next_state in self._cells:
                    self.graph.add_edge(state, next_state)

    def draw_graph(self):
        '''
            Draws the graph of the maze
        '''
        nx.draw(self.graph, pos = nx.nx_pydot.graphviz_layout(self.graph), node_size=50, 
        node_color='lightblue', linewidths=0.25, with_labels=False)


# ---------------------------- create/edit layouts --------------------------- #

def layout_from_image(image, final_size=200):
    '''
        Loads an image from file, thresholds it and downscales it to be of size
        final_size x final_size
    '''
    image = Image.open(image)
    image.thumbnail((final_size, final_size), Image.ANTIALIAS)
    layout = np.array(image)[:, :, 0]
    layout[layout<=50] = 0
    layout[layout>50] = 1
    return 1 - layout

def add_platform(layout, x, y, side=4):
    '''
        Add a  platform to a maze layout.

        Arguments:
            layout: np.array storing maze layout
            x, y: int. Coordinates of platform center
            side: int. Width/height of platform
    '''
    h = int(side/2)
    layout[x-h:x+h, y-h:y+h] = 0
    return layout

def add_horizontal_bridge(layout, x0, x1, y, double=False):
    '''
        Add an horizontal  to a maze layout.

        Arguments:
            layout: np.array storing maze layout
            x0, x1: int.coordinates of start and end points of bridge
            y: int. Y coordinates of bridge
    '''
    layout[y, x0:x1] = 0
    if double:
        layout[y+1, x0:x1] = 0
    return layout

def add_vertical_bridge(layout, x, y0, y1):
    '''
        Add a vertical bridge to a maze layout.

        Arguments:
            layout: np.array storing maze layout
            x: int. X coordinates of bridge
            y0, y1: int. Start and end coordinates of bridge
    '''
    layout[y0:y1, x] = 0
    return layout

def add_diagonal_bridge(layout, x0, x1, y0, y1, double=False):
    '''
        Add a diagonal bridge to a maze layout.

        Arguments:
            layout: np.array storing maze layout
            x0, x1: int. X coordinates of start and end points
            y0, y1: int. Y coordinates of start and end points
            double: bool. If true the bridge has double thickness
    '''
    for t in np.linspace(0, 1, 100):
        '''
        Add a   to a maze layout.

        Arguments:
            layout: np.array storing maze layout
        '''
        x = int((1-t)*x0 + t*x1)
        y = int((1-t)*y0 + t*y1)
        layout[y, x] = 0

        if double:
            layout[y-1, x] = 0
    return layout

def add_catwalk(layout, x, y, height=3):
    '''
        Add a threat platform catwalk to a maze layout.

        Arguments:
            layout: np.array storing maze layout
            x,y: int start coordinates of catwalk
            height: int. 'lengt' of the catwalk
    '''
    layout[y:y+height, x] = 0
    return layout

def geodist(maze, shelter):
    """[Calculates the geodesic distance from the shelter at each location of the maze]
	
	Arguments:
		maze {[np.ndarray]} -- [maze as 2d array]
		shelter {[np.ndarray]} -- [coordinates of the shelter]
	"""
    maze = 1 - maze.copy()
    phi = np.ones_like(maze)
    mask = maze == 0
    masked_maze = np.ma.MaskedArray(phi, mask)

    masked_maze[shelter[1], shelter[0]] = 0

    distance_from_shelter = np.array(skfmm.distance(masked_maze))
    return distance_from_shelter