# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import paper
from paper import paths
import os

from paper.dbase.TablesDefinitionsV4 import Explorations, Session
from paper.dbase.utils import convert_roi_id_to_tag, plot_rois_positions
from paper.paths import plots_dir
from paper import maze_colors
from paper.helpers.mazes_stats import get_mazes

from fcutils.plotting.utils import clean_axes, save_figure
from fcutils.plotting.plot_distributions import plot_kde
from fcutils.plotting.colors import *


# %%
def get_maze_explorations(maze_design, naive=None, lights=1):
    # Get all explorations for a given maze
    query = Explorations * Session * Session.Metadata * Session.Shelter  - 'experiment_name="Foraging"' 
    
    query = (query & "maze_type={}".format(maze_design))
    if naive is not None: query = (query & "naive={}".format(naive))
    if lights is not None: query = (query & "lights={}".format(lights))

    return pd.DataFrame(query.fetch())

def get_session_time_on_arms(exploration, normalize = False, maze=None):
    """

        Given one entry of Explorations() this returns the time spent on each
        of the arms during the exploration. If normalise=True, the time per arm
        is normalized by the relative length of each arm (relative to the right path)
    """
    tracking = exploration.body_tracking

    # Get time spent on each path in seconds normalized by total duration of explorations
    on_left = len(tracking[tracking[:, 0]  < 450])  / exploration.fps
    on_right = len(tracking[tracking[:, 0]  > 550])  / exploration.fps
    tot = on_left + on_right
    if not tot: return None
    on_left /= tot
    on_right /= tot


    # NOrmalize
    if normalize: 
        metadata = get_mazes()[f'maze{maze}']
        ratio = metadata['right_path_length'] / metadata['left_path_length']
        on_left *= ratio

    return on_left, on_right



# %%
mazes = [1, 2, 3, 4, 6]
NORMALIZE = True


f, axarr = plt.subplots(ncols = len(mazes), figsize=(30, 8))

for maze, ax in zip(mazes, axarr):
    explorations = get_maze_explorations(maze,  naive=None, lights=1)

    occupancy = []

    for i, exp in tqdm(explorations.iterrows()):
        occ = get_session_time_on_arms(exp, normalize=NORMALIZE, maze=maze)
        if occ is None: continue
        occupancy.append(occ)

    occupancy = pd.DataFrame(dict(
        l = [l for l, r in occupancy],
        r = [r for l, r in occupancy]
    ))



    ax.plot(occupancy.T, 'o-',  ms=10, color=[.7, .7, .7])
    plot_kde(data=occupancy.l, vertical=True, normto=-.2, ax=ax, color=maze_colors[f'maze{maze}'], alpha=.1)
    plot_kde(data=occupancy.r, z=1, vertical=True, normto=.2, ax=ax, color=maze_colors[f'maze{maze}'], alpha=.1)
    ax.plot([0, 1], occupancy.median(), '-o', ms=25, color=maze_colors[f'maze{maze}'], lw=3)

    ax.set(title=f'Maze {maze}')
clean_axes(f)

save_figure(f, os.path.join(plots_dir, f"exploration_arm_occupancy{'_norm' if NORMALIZE else ''}"))

# %%
# %%
