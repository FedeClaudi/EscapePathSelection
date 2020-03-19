# Datajoint
import djconn
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
    'm0': darkgreen,
    'm1': palette[0],
    'm1-dark': darkred, 
    'm1-light': red, 
    'm2': palette[1],
    'm3': palette[2],
    'm4': palette[3],
    'm6': salmon,
    'mb': palette[4],
    'mb1': palette[4],
    'mb2': palette[5]
}

palette = makePalette(teal, lilla, 4, False)
arms_colors = {
    "left": teal,
    'center': darkcyan,
    "right": lilla,
}



psychometric_mazes = ["m1", "m2", "m3", "m4"]
psychometric_mazes_and_dark = ["m1", "m2", "m3", "m4", "m1-dark"]
five_mazes = ["m1", "m2", "m3", "m4", "m6"]
m6 = ["m6"]
m0 = ["m0"]
allmazes = ["m1", "m2", "m3", "m4", "m6", "mb"]
arms = ['left', 'right', 'center']