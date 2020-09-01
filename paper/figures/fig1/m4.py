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

from paper.figures.fig1 import savedir

# %%

# ---------------------------------------------------------------------------- #
#                                   LOAD DATA                                  #
# ---------------------------------------------------------------------------- #

print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'all',
    escapes_dur=False,
)


trials =  TrialsLoader(**params).load_trials_by_condition(maze_design=4)

# Load explorations for each session and check that the mouse goes on the central arm
explorations = get_maze_explorations(4)
f, axarr = plt.subplots(ncols=5, nrows=5, figsize=(16, 16))
axarr = axarr.flatten()

keep_sessions = [8, 9, 11, 12, 13, 14]
keep_session_names = []

for ax, (i, expl) in zip(axarr, explorations.iterrows()):
    if i in keep_sessions:
        color = green
        keep_session_names.append(expl.session_name)
    else:
        color = [.6, .6, .6]
    ax.plot(expl.body_tracking[:, 0], expl.body_tracking[:, 1], lw=.5, color=color)
    ax.set(title=f'{i} - {expl.session_name}')



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


# %%

# ---------------------------------------------------------------------------- #
#                                   PLOTTING                                   #
# ---------------------------------------------------------------------------- #
cmaps = ['Blues', 'Reds', 'Greens']
arms = ['Left', 'Center', 'Right']
colors = [darkseagreen, salmon, lightseagreen]



# %%
"""
    Plot example trials for M4

    Example of individual trials tracking on the threat platform
"""

COL_SPEED = False

left_trials_idx = [i for i,t in trials.iterrows() if t.escape_arm == 'left'][1]
right_trials_idx = [i for i,t in trials.iterrows() if t.escape_arm == 'right'][7]
center_trials_idx = [i for i,t in trials.iterrows() if t.escape_arm == 'center'][1]
idxs = [left_trials_idx, center_trials_idx, right_trials_idx]

f, ax = plt.subplots(figsize=(9, 9))

for arm, tidx, color, cmap in zip(arms, idxs, colors, cmaps):
    trial = trials.loc[trials.index == tidx].iloc[0]
    plot_trial_tracking_as_lines(trial, ax, color, 10, 
            color_by_speed=COL_SPEED, cmap=cmap, 
            outline_color=[.4, .4, .4], 
            thin_alpha=.5, thin_lw=2)

    # ax.set(title=arm, xlim=[430, 560], ylim=[220, 350])
    ax.axis('off')

name = 'colored by speed' if COL_SPEED else ''
save_figure(f, os.path.join(savedir, 'M4 sing trials example ' + name), svg=True)

# %%
"""
    Individual mean trials color coded by speed
"""
f, ax = plt.subplots(figsize=(9, 9))

L, R, C = dict(x=[], y=[], s=[]), dict(x=[], y=[], s=[]), dict(x=[], y=[], s=[]) # store the speed profiles of escapes by arm


for i, trial in trials.iterrows():
    if trial.escape_arm == 'left':
        L['s'].append(trial.body_speed)
        L['y'].append(trial.body_xy[:, 1])
        L['x'].append(trial.body_xy[:, 0])
        color = colors[0]
    elif trial.escape_arm == 'right':
        if trial.body_xy[:, 0].min() < 400: continue
        R['s'].append(trial.body_speed)
        R['y'].append(trial.body_xy[:, 1])
        R['x'].append(trial.body_xy[:, 0])
        color = colors[2]

    else:
        C['s'].append(trial.body_speed)
        C['y'].append(trial.body_xy[:, 1])
        C['x'].append(trial.body_xy[:, 0])
        color = colors[1]

    ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=desaturate_color(color), lw=.75)

print(f'{len(L["s"])} left trials, {len(R["s"])} right and {len(C["s"])} left')


# Normalize escape duration
L_std = {k:resample_list_of_arrayes_to_avg_len(v).std(0) for k,v in L.items()}
R_std = {k:resample_list_of_arrayes_to_avg_len(v).std(0) for k,v in R.items()}
C_std = {k:resample_list_of_arrayes_to_avg_len(v).std(0) for k,v in C.items()}

L = {k:resample_list_of_arrayes_to_avg_len(v).mean(0) for k,v in L.items()}
R = {k:resample_list_of_arrayes_to_avg_len(v).mean(0) for k,v in R.items()}
C = {k:resample_list_of_arrayes_to_avg_len(v).mean(0) for k,v in C.items()}

for arm, std_data, data, color in zip(arms, [L_std, C_std, R_std], [L, C, R], colors):
    x = line_smoother(data['x'][10:-2], window_size=11)
    stdx = line_smoother(std_data['x'][10:-2], window_size=11)
    y = line_smoother(data['y'][10:-2], window_size=11)
    stdy = line_smoother(std_data['y'][10:-2], window_size=11)

    a = x - stdx
    c = x + stdx
    ax.fill_betweenx(y, a, c, color=color, alpha=.2)
    # ax.fill_between(x, c, a, color=color, alpha=.2)


    ax.plot(x, y, color=[.4, .4, .4], lw=7)
    ax.plot(x, y, color=color, lw=6)

ax.axis('off')
_ = ax.set(title='Mean trajectory per arm colored by speed')
save_figure(f, os.path.join(savedir, 'F1 M4 tracking'), svg=True)
