# Datajoint
from paper import djconn
dbname, schema = djconn.start_connection()

# Paths
from paper import paths


# Matplotlib config
import paper.utils.mpl_config


# Colors
from fcutils.plotting.colors import *
from fcutils.plotting.colors import makePalette


palette = makePalette(green, orange, 7 , False)
maze_colors = {
    'maze0': darkgreen,
    'maze1': palette[0],
    'maze1-dark': darkred, 
    'maze1-light': red, 
    'maze2': palette[1],
    'maze3': palette[2],
    'maze4': palette[3],
    'maze6': salmon,
    'mb': palette[4],
    'mb1': palette[4],
    'mb2': palette[5]
}

palette = makePalette(teal, lilla, 4, False)
arms_colors = {
    "left": seagreen,
    'center': salmon,
    "right": darkseagreen,
}



psychometric_mazes = ["maze1", "maze2", "maze3", "maze4"]
psychometric_mazes_and_dark = ["maze1", "maze2", "maze3", "maze4", "maze1-dark"]
five_mazes = ["maze1", "maze2", "maze3", "maze4", "maze6"]
m6 = ["maze6"]
m0 = ["maze0"]
allmazes = ["maze1", "maze2", "maze3", "maze4", "maze6", "mb"]
arms = ['left', 'right', 'center']