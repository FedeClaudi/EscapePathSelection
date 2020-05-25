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

from behaviour.utilities.signals import get_times_signal_high_and_low


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
"""
    Plot the time spent on each path for each exploration
"""
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

"""
    Plot the fraction fo IN/OUT trips between the left and right path for each maze
"""

f, ax = plt.subplots()

only = 'in'

for n, maze in enumerate(mazes):
    Y = [0]
    explorations = get_maze_explorations(maze,  naive=None, lights=1)
    for i, e in explorations.iterrows():

        # Get left and right outwards and inwards trips
        tracking = e.body_tracking

        left = np.zeros_like(tracking[:, 0])
        left[tracking[:, 0] < 450] = 1
        l_starts, l_ends = get_times_signal_high_and_low(left, th=.5)

        right = np.zeros_like(tracking[:, 0])
        right[tracking[:, 0] > 550] = 1
        r_starts, r_ends = get_times_signal_high_and_low(right, th=.5)

        n_trips = dict(
            left = dict(outward = 0, inward = 0),
            right = dict(outward = 0, inward = 0),

        )
        
        for start, end in zip(l_starts, l_ends):
            if end - start < 100: continue
            if tracking[start, 1]  > 500:
                n_trips['left']['outward'] += 1
            else:
                n_trips['left']['inward'] += 1

        
        for start, end in zip(r_starts, r_ends):
            if end - start < 100: continue
            if tracking[start, 1]  > 500:
                n_trips['right']['outward'] += 1
            else:
                n_trips['right']['inward'] += 1

        # Summarise
        if only is None: # consider all trips
            L, R = n_trips['left']['outward']+n_trips['left']['inward'], n_trips['right']['outward']+n_trips['right']['inward']
        elif only == 'out':
            L, R = n_trips['left']['outward'], n_trips['right']['outward']
        elif only == 'in':
            L, R = n_trips['left']['inward'], n_trips['right']['inward']
        else:
            raise ValueError

        if not L and not R:
            continue
        elif not L:
            y = 1
        elif not R:
            y = 0
        else:
            y = R/(L+R)
        
        # Plot scatter
        ax.scatter(np.random.normal(n, 0.001), y, s=40, color=[.4, .4, .4], alpha=.5)
        Y.append(y)

    # Plot KDE and mean
    plot_kde(data=Y, vertical=True,  z=n, normto=.3, ax=ax, color=maze_colors[f'maze{maze}'], alpha=.1, kde_kwargs=dict(bw=.02))
    ax.scatter(n, np.mean(Y), s=160, color='red', zorder=99)


ax.axhline(0.5, lw=2, ls='--', color='k')

ax.set(ylim=[-.1, 1.1], title=f'Fraction of {only} trips per maze [exploration]')

clean_axes(f)



save_figure(f, os.path.join(plots_dir, f"exploration_trips_fraction{only}"))

# %%
