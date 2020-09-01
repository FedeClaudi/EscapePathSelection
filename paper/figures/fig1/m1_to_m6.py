# %%
"""
    Functions to create the plots in Figure 1 - M4 dead end 

"""
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt
from matplotlib.patches import Ellipse
from scipy.signal import resample

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point, plot_shaded_withline, plot_mean_and_error
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.colors import *
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as calc_angle
from fcutils.maths.filtering import line_smoother

import sys
sys.path.append(r'C:\Users\Federico\Documents\GitHub\EscapePathSelection')

import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.utils.explorations import get_maze_explorations
from paper.utils.misc import resample_list_of_arrayes_to_avg_len, plot_trial_tracking_as_lines
from paper.helpers.mazes_stats import get_mazes

from paper.figures.fig1 import savedir
# %%
# %%
# --------------------------------- Load data -------------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = 1, 
    tracking = 'all',
    catwalk = True,
)

trials = TrialsLoader(**params)
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials

# --------------------------- P(R) and other stuff --------------------------- #
print("Grouped bayes")
hits, ntrials, p_r, n_mice, trs = trials.get_binary_trials_per_dataset()

# Grouped bayes
grouped_pRs = trials.grouped_bayes_by_dataset_analytical()

# Individuals hierarchical bayes
print("Hierarchical bayes")
cache_path = os.path.join(paths.cache_dir, 'psychometric_hierarchical_bayes.h5')
try:
    hierarchical_pRs = pd.read_hdf(cache_path)
except Exception as e:
    print(f"Couldnt load because of {e}")
    hierarchical_pRs = trials.individuals_bayes_by_dataset_hierarchical()
    hierarchical_pRs.to_hdf(cache_path, key='hdf')

# ----------------------------- Get mazes metadata ---------------------------- #
print("Mazes stats")
_mazes = get_mazes()


# %%
cmaps = ['Blues', 'Reds', 'Greens']
arms = ['Left', 'Right']
colors = [darkseagreen, lightseagreen]

# --------------------------------- PLOTTING --------------------------------- #

# %%
"""
    Plot example trials for each psycometric maze
"""

COL_SPEED = False


f, axarr = plt.subplots(figsize=(9*5, 9), ncols=5, sharex=True, sharey=True)

IDXS = dict(
    maze1 = [2, 7],
    maze2 = [1, 7],
    maze3 = [2, 7],
    maze4 = [2, 7],
    maze6 = [2, 7],

)

for ax, (maze, trs) in zip(axarr, trials.datasets.items()):
    catwalks = []
    for i, trial in trs.iterrows():
        if trial.body_xy[0, 1] > 230:
            catwalks.append(0)
        else:
            catwalks.append(1)
    trs['catwalk'] = catwalks
    trs = trs.loc[trs.catwalk == 1]


    left_trials_idx = [i for i,t in trs.iterrows() if t.escape_arm == 'left'][IDXS[maze][0]]
    right_trials_idx = [i for i,t in trs.iterrows() if t.escape_arm == 'right'][IDXS[maze][1]]
    idxs = [left_trials_idx, right_trials_idx]

    col = paper.maze_colors[maze]

    for arm, tidx in zip(arms, idxs):
        if 'Left' in arm:
            color = col
        else:
            color = desaturate_color(col)


        trial = trs.loc[trs.index == tidx].iloc[0]
        plot_trial_tracking_as_lines(trial, ax, color, 15, 
                thick_lw=6,
                color_by_speed=COL_SPEED, 
                outline_color=[.4, .4, .4], 
                thin_alpha=.25, thin_lw=2)

        # ax.set(title=arm, xlim=[430, 560], ylim=[220, 350])
        ax.axis('off')

clean_axes(f)
plt.tight_layout()

name = 'colored by speed' if COL_SPEED else ''
save_figure(f, os.path.join(savedir, 'M1-M6 sing trials example ' + name), svg=True)
# %%
"""
    Plot all trials from psychometric mazes
"""

f, axarr = plt.subplots(figsize=(9*5, 9), ncols=5, sharex=True, sharey=True)

IDXS = dict(
    maze1 = [2, 7],
    maze2 = [1, 7],
    maze3 = [2, 7],
    maze4 = [2, 7],
    maze6 = [2, 7],

)


for ax, (maze, trs) in zip(axarr, trials.datasets.items()):
    L, R = dict(x=[], y=[], s=[]), dict(x=[], y=[], s=[]) # store the speed profiles of escapes by arm


    for i, trial in trs.iterrows():
        if trial.escape_arm == 'left':
            color = paper.maze_colors[maze]


            L['s'].append(trial.body_speed)
            L['y'].append(trial.body_xy[:, 1])
            L['x'].append(trial.body_xy[:, 0])
        else:

            color = desaturate_color(paper.maze_colors[maze])
            R['s'].append(trial.body_speed)
            R['y'].append(trial.body_xy[:, 1])
            R['x'].append(trial.body_xy[:, 0])

        ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1],
                    color=desaturate_color(color), lw=.5,  alpha=.4)

        ax.axis('off')


    L = {k:resample_list_of_arrayes_to_avg_len(v).mean(0) for k,v in L.items()}
    R = {k:resample_list_of_arrayes_to_avg_len(v).mean(0) for k,v in R.items()}
    colors = [paper.maze_colors[maze], desaturate_color(paper.maze_colors[maze])]

    for arm, data, color in zip(arms, [L,  R], colors):
        x = line_smoother(data['x'][10:-2], window_size=11)
        y = line_smoother(data['y'][10:-2], window_size=11)

        ax.plot(x, y, color='w', lw=7, zorder=99)
        ax.plot(x, y, color=color, lw=6, zorder=99)

save_figure(f, os.path.join(savedir, 'M1-M6 all trials'), svg=True)
