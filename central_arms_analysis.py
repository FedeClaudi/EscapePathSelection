# %%
# Imports
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.colors import *
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.utils.explorations import get_maze_explorations

# %%

# ---------------------------------------------------------------------------- #
#                                     MAZE0                                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Load data -------------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'all'
)

trials =  TrialsLoader(**params).load_trials_by_condition(maze_design=0)
trials.head()

# %%

# --------------------------- Plot trials tracking --------------------------- #
f, axarr = plt.subplots(ncols=2, figsize=(16, 9))

for i, trial in trials.iterrows():
    axarr[0].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=[.6, .6, .6], lw=.5)



left = trials.loc[trials.escape_arm == 'left']
center = trials.loc[trials.escape_arm == 'center']
right = trials.loc[trials.escape_arm == 'right']

colors = [darkseagreen, salmon, lightseagreen]

for trial, color in zip([left, center, right], colors):
    axarr[0].plot(trial.iloc[0].body_xy[:, 0], trial.iloc[0].body_xy[:, 1], color=white, lw=8)
    axarr[0].plot(trial.iloc[0].body_xy[:, 0], trial.iloc[0].body_xy[:, 1], color=color, lw=7)


# ---------------------- Plot escape probability per arm --------------------- #
prs = [len(t)/len(trials) for t in [left, center, right]]
axarr[1].bar([0, 1, 2], prs, color=colors)

axarr[0].axis('off')
axarr[0].set(title=f'Maze0 - n trials: {len(trials)}')
axarr[1].set(title='probability of escape per arm', ylim=[0, 1], xticks=[0, 1, 2], xticklabels=['left', 'center', 'right'], ylabel='p(arm)', xlabel='arm')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'maze0_p_arm'), svg=False)


# %%
# ---------------------------------------------------------------------------- #
#                              MAZE 4 CENTRAL ARM                              #
# ---------------------------------------------------------------------------- #
# ! check that I'm only taking sessions that actually have the central arm in the maze


print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'all',
    escapes_dur=False,
)


trials =  TrialsLoader(**params).load_trials_by_condition(maze_design=4)

# %%
# Load explorations for each session and check that the mouse goes on the central arm
explorations = get_maze_explorations(4)
f, axarr = plt.subplots(ncols=5, nrows=5)
axarr = axarr.flatten()

keep_sessions = [9, 10, 12, 13, 14, 15]
keep_session_names = []

for ax, (i, expl) in zip(axarr, explorations.iterrows()):
    if i in keep_sessions:
        color = green
        keep_session_names.append(expl.session_name)
    else:
        color = [.6, .6, .6]
    ax.plot(expl.body_tracking[:, 0], expl.body_tracking[:, 1], lw=.5, color=color)
    ax.set(title=f'{i} - {expl.session_name}')


# %%
print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'all',
    escapes_dur=True,
)

tloader = TrialsLoader(**params)
tloader.max_duration_th = 15
trials =  tloader.load_trials_by_condition(maze_design=4)
trials = trials.loc[trials.session_name.isin(keep_session_names)]

# --------------------------- Plot trials tracking --------------------------- #
f, axarr = plt.subplots(ncols=2, figsize=(16, 9))

for i, trial in trials.iterrows():
    axarr[0].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=[.6, .6, .6], lw=.5)



left = trials.loc[trials.escape_arm == 'left']
center = trials.loc[trials.escape_arm == 'center']
right = trials.loc[trials.escape_arm == 'right']

colors = [darkseagreen, salmon, lightseagreen]

for trial, color in zip([left, center, right], colors):
    axarr[0].plot(trial.iloc[0].body_xy[:, 0], trial.iloc[0].body_xy[:, 1], color=white, lw=8)
    axarr[0].plot(trial.iloc[0].body_xy[:, 0], trial.iloc[0].body_xy[:, 1], color=color, lw=7)


# ---------------------- Plot escape probability per arm --------------------- #
prs = [len(t)/len(trials) for t in [left, center, right]]
axarr[1].bar([0, 1, 2], prs, color=colors)

axarr[0].axis('off')
axarr[0].set(title=f'Maze0 - n trials: {len(trials)}')
axarr[1].set(title='probability of escape per arm', ylim=[0, 1], xticks=[0, 1, 2], xticklabels=['left', 'center', 'right'], ylabel='p(arm)', xlabel='arm')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'maze4_p_arm_all_trials'), svg=False)



# %%