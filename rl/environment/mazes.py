import numpy as np

from rl.environment.maze import (
    Maze,
    layout_from_image,
    add_platform,
    add_horizontal_bridge,
    add_vertical_bridge,
    add_diagonal_bridge,
    add_catwalk,
)

# ---------------------------------------------------------------------------- #
#                                     MAZES                                    #
# ---------------------------------------------------------------------------- #

# ------------------------ Based on experimental mazes ----------------------- #

class PsychometricM1(Maze):
    def __init__(self, *args, **kwargs):
        # layout = layout_from_image('./maze_images/M1.png')
        layout = np.ones((50, 50))

        # add platforms
        platforms = [(25, 16), (33, 24), (25, 33), (9, 16)]
        for (px, py) in platforms:
            layout = add_platform(layout, py, px)

        # add bridges
        layout = add_horizontal_bridge(layout, 23, 27, 30, double=False)

        layout = add_horizontal_bridge(layout, 9, 25, 15, double=False)
        layout = add_diagonal_bridge(layout, 9, 25, 16, 33, double=False)
        layout = add_diagonal_bridge(layout, 25, 33, 33, 24, double=False)
        layout = add_catwalk(layout, 24, 33, 5)

        layout = add_diagonal_bridge(layout, 25, 33, 16, 24, double=True)
        # layout = add_diagonal_bridge(layout, 24, 32, 16, 24, double=True)

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(24, 34),
            SHELTER=(24, 15),
            MIN_N_STEPS=28,
            description='Asymmetric M1 maze from psychometric experiments',
            name='PsychometricM1',
            **kwargs,
        )

class PsychometricM2(Maze):
    def __init__(self, *args, **kwargs):
        # layout = layout_from_image('./maze_images/M2.png')
        layout = np.ones((50, 50))

        # add platforms
        platforms = [(25, 16), (33, 24), (25, 33), (13, 20)]
        for (px, py) in platforms:
            layout = add_platform(layout, py, px)

        # add bridges
        layout = add_diagonal_bridge(layout, 13, 25, 20, 16, double=True)
        layout = add_diagonal_bridge(layout, 25, 33, 16, 24, double=True)
        layout = add_diagonal_bridge(layout, 13, 25, 20, 33)
        layout = add_diagonal_bridge(layout, 25, 33, 33, 24)
        layout = add_catwalk(layout, 24, 33, 5)

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(24, 34),
            SHELTER=(24, 15),
            MIN_N_STEPS=28,
            description='Asymmetric M1 maze from psychometric experiments',
            name='PsychometricM2',
            **kwargs,
        )

class PsychometricM6(Maze):
    def __init__(self, *args, **kwargs):
        # layout = layout_from_image('./maze_images/M6.png')
        layout = np.ones((50, 50))

        # add platforms
        platforms = [(25, 16), (42, 33), (25, 33), (9, 16)]
        for (px, py) in platforms:
            layout = add_platform(layout, py, px)

        # add bridges
        layout = add_horizontal_bridge(layout, 9, 25, 15)
        layout = add_horizontal_bridge(layout, 9, 25, 16)
        layout = add_diagonal_bridge(layout, 25, 42, 16, 33, double=True)
        layout = add_diagonal_bridge(layout, 25, 42, 15, 32, double=True)
        layout = add_diagonal_bridge(layout, 9, 25, 16, 33, double=True)
        layout = add_horizontal_bridge(layout, 25, 42, 32)
        layout = add_horizontal_bridge(layout, 25, 42, 31)
        layout = add_catwalk(layout, 24, 33, 5)


        Maze.__init__(
            self, 
            layout,
            *args,
            START=(24, 34),
            SHELTER=(24, 15),
            MIN_N_STEPS=28,
            description='Asymmetric M1 maze from psychometric experiments',
            name='PsychometricM6',
            **kwargs,
        )

class MB(Maze):
    def __init__(self, *args, **kwargs):
        # layout = layout_from_image('./maze_images/MB.png')
        # self._blocked_layout = layout_from_image('./maze_images/MB_blocked.png')
        self.tracking_scaling_factor = 44 / 1000  # maze images are 50 wide and tracking is from images 1000 wide
        self.tracking_x_shift = 1
        self.tracking_y_shift = 8
        layout = np.ones((50, 50))

        # add platforms
        platforms = [
            (10, 16), (25, 16), (40, 16),
            (25, 28),
            (10, 40), (25, 40), (40, 40)
        ]
        for (px, py) in platforms:
            layout = add_platform(layout, py, px)
        
        # add bridges
        layout = add_horizontal_bridge(layout, 10, 40, 16)
        layout = add_horizontal_bridge(layout, 10, 40, 40)
        layout = add_vertical_bridge(layout, 10, 16, 40)
        layout = add_vertical_bridge(layout, 40, 16, 40)

        layout = add_diagonal_bridge(layout, 25, 30, 40, 33)
        layout = add_diagonal_bridge(layout, 25, 20, 40, 33)
        layout = add_diagonal_bridge(layout, 20, 25, 33, 28, double=True)
        layout = add_diagonal_bridge(layout, 30, 25, 33, 28, double=True)

        layout = add_catwalk(layout, 25, 40, 5)
        layout = add_catwalk(layout, 24, 40, 5)

        # get blocked and add final bridge
        self._blocked_layout = layout.copy()
        layout = add_vertical_bridge(layout.copy(), 25, 16, 28)
        layout = add_vertical_bridge(layout, 24, 16, 28)

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(24, 44),
            SHELTER=(25, 16),
            MIN_N_STEPS=28,
            description='Very lagre maze',
            name='MB',
            **kwargs,
        )





# ------------------------------ simulted mazes ------------------------------ #
class Open(Maze):
    def __init__(self, *args, **kwargs):
        N = 60
        layout = np.zeros((N, N))

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(N-1, N-1),
            SHELTER=(0, 0),
            MIN_N_STEPS=30,
            description='Open arena',
            name='Open',
            **kwargs,
        )

class Wall(Maze):
    def __init__(self, *args, **kwargs):
        N = 15
        layout = np.zeros((N, N))
        layout[7, 2:-2] = 1
        layout[8, 2:-2] = 1

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(8, 14),
            SHELTER=(8, 0),
            MIN_N_STEPS=30,
            description='Wall',
            name='Wall',
            **kwargs,
        )

class M0(Maze):
    def __init__(self, *args, **kwargs):
        layout = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(1, 4),
            SHELTER=(1, 0),
            MIN_N_STEPS=6,
            description='square maze with offset start/shelter to be asymmetric',
            name='M0',
            **kwargs,
        )

class M1(Maze):
    def __init__(self, *args, **kwargs):
        layout = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0],
        ])

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(2, 4),
            SHELTER=(2, 0),
            MIN_N_STEPS=8,
            description='length symmetric, euclidean asymmetric',
            name='M1',
            **kwargs,
        )

class M2(Maze):
    def __init__(self, *args, **kwargs):
        layout = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ])

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(2, 4),
            SHELTER=(2, 0),
            MIN_N_STEPS=8,
            description='asymmetri with blocked path',
            name='M2',
            **kwargs,
        )

        # store a blocked layout
        self._blocked_layout = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ])

class M3(Maze):
    def __init__(self, *args, **kwargs):
        layout = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ])

        Maze.__init__(
            self, 
            layout,
            *args,
            START=(2, 4),
            SHELTER=(2, 0),
            MIN_N_STEPS=8,
            description='shortcut',
            name='M3',
            **kwargs,
        )

        # store a shortcut layout
        self._shortcut_layout = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ])
