# %%
# Imports
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt
from scipy.signal import resample
import matplotlib.colors  

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.colors import *
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point, plot_shaded_withline
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.utils.explorations import get_maze_explorations
from paper.utils.misc import resample_list_of_arrayes_to_avg_len, plot_trial_tracking_as_lines


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
def plot_trials(maze, trials):
    f, axarr = plt.subplots(ncols=2, figsize=(16, 9))

    tracking = {'left': {'x':[], 'y':[]}, 'center':{'x':[], 'y':[]}, 'right':{'x':[], 'y':[]}}

    for i, trial in trials.iterrows():
        axarr[0].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=[.6, .6, .6], lw=.5)

        tracking[trial.escape_arm]['x'].append(trial.body_xy[:, 0])
        tracking[trial.escape_arm]['y'].append(trial.body_xy[:, 1])

    # Get average trajectory for each arm
    colors = [darkseagreen, salmon, lightseagreen]
    for (arm, data), color in zip(tracking.items(), colors):
        mean_dur = np.mean([len(X) for X in data['x']]).astype(np.int32)
        x = np.nanmedian(np.vstack([resample(X, mean_dur) for X in data['x']]), 0)
        y = np.nanmedian(np.vstack([resample(X, mean_dur) for X in data['y']]), 0)

        axarr[0].plot(x[2:-2], y[2:-2], color=white, lw=8)
        axarr[0].plot(x[2:-2], y[2:-2], color=color, lw=7)


    # ---------------------- Plot escape probability per arm --------------------- #
    left = trials.loc[trials.escape_arm == 'left']
    center = trials.loc[trials.escape_arm == 'center']
    right = trials.loc[trials.escape_arm == 'right']
    prs = [len(t)/len(trials) for t in [left, center, right]]
    axarr[1].bar([0, 1, 2], prs, color=colors)

    axarr[0].axis('off')
    axarr[0].set(title=f'{maze} - n trials: {len(trials)}')
    axarr[1].set(title='probability of escape per arm', ylim=[0, 1], xticks=[0, 1, 2], 
                xticklabels=['left', 'center', 'right'], ylabel='p(arm)', xlabel='arm')

    clean_axes(f)
    save_figure(f, os.path.join(paths.plots_dir, f'{maze}_p_arm'), svg=True)


# %%
plot_trials('maze0', trials)

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
plot_trials('maze4', trials)



# %%
"""
    Look at arm occupancy during the exploration
"""
colors = [darkseagreen, salmon, lightseagreen]

def get_session_time_on_arms(exploration):
    """

        Given one entry of Explorations() this returns the time spent on each
        of the arms during the exploration. If normalise=True, the time per arm
        is normalized by the relative length of each arm (relative to the right path)
    """
    x, y = exploration.body_tracking[:, 0], exploration.body_tracking[:, 1]

    # Get time spent on each path in seconds normalized by total duration of explorations
    on_left = len(x[x  < 450])  / exploration.fps
    on_right = len(x[x  > 550])  / exploration.fps

    on_center = np.where((x > 450) & (x < 550) &
                        (y > 380) & (y < 520))
    # raise ValueError(on_center[0])
    on_center = len(x[on_center[0]]) / exploration.fps
    return on_left, on_right, on_center

f, ax = plt.subplots()
for i, expl in explorations.iterrows():
    if i not in keep_sessions: continue
    
    l, r, c = get_session_time_on_arms(expl)

    ax.scatter([0, 1, 2], [l, c, r], c=colors, zorder=99, s=159)
    ax.plot([0, 1, 2], [l, c, r], color=[.8, .8, .8])

ax.set(title='Time spent on paths during exploration. M4 - dead end',
        ylabel='TOT time (s)', 
        xticks=[0, 1, 2], xticklabels=['left', 'center', 'right'])
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'M4 deaded end arm occupancy exploration'), svg=True)

# %%
"""
    Plot example tracking for M4 with dead end arm looking at speed etc
"""
colors = [darkseagreen, salmon, lightseagreen]
arms = ['Left', 'Center', 'Right']

L, R, C = [], [], [] # store the speed profiles of escapes by arm
at_center, lengths = [], []
for i, trial in trials.iterrows():
    if trial.escape_arm == 'left':
        L.append(trial.body_speed)
        lengths.append(len(trial.body_speed))
    elif trial.escape_arm == 'right':
        R.append(trial.body_speed)
        lengths.append(len(trial.body_speed))
    else:
        x, y = trial.body_xy[:, 0], trial.body_xy[:, 1]
        at_cent = np.where((x > 460) & (x < 540) & (y > 450) & (y < 550))[0][0]
        at_center.append(at_cent)

        C.append(trial.body_speed)


# Normalize escape duration
L = resample_list_of_arrayes_to_avg_len(L, N=100).mean(0)
R = resample_list_of_arrayes_to_avg_len(R, N=100).mean(0)
# C = resample_list_of_arrayes_to_avg_len(C, N=int(np.mean(at_center) * 100 / np.mean(lengths))).mean(0)
C = resample_list_of_arrayes_to_avg_len(C, N=100).mean(0)

f, ax = plt.subplots(figsize=(16, 9))
for arm, speed, color in zip(arms, [L, C, R], colors):
    # ax.plot(speed, color=color, lw=2, label=arm)
    plot_shaded_withline(ax, np.arange(len(speed)), speed, color=color, label=arm, lw=4, alpha=.05)

ax.legend() 
ax.set(title='M4 dead end | escape speed by arm ',
        xticks=[0, 100], xticklabels=['start', 'end'],
        ylabel='speed')
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'M4 deaded end escape speed by arm'), svg=True)

# %%
"""
    Similar to above, looking at running speed
    for escapes of each arm, but including tracking
"""
cmaps = ['Blues', 'Reds', 'Greens']
arms = ['Left', 'Center', 'Right']

L, R, C = dict(x=[], y=[], s=[]), dict(x=[], y=[], s=[]), dict(x=[], y=[], s=[]) # store the speed profiles of escapes by arm
for i, trial in trials.iterrows():
    if trial.escape_arm == 'left':
        L['s'].append(trial.body_speed)
        L['y'].append(trial.body_xy[:, 1])
        L['x'].append(trial.body_xy[:, 0])
    elif trial.escape_arm == 'right':
        R['s'].append(trial.body_speed)
        R['y'].append(trial.body_xy[:, 1])
        R['x'].append(trial.body_xy[:, 0])
    else:
        # Get when the mouse reaches center arm end
        x, y = trial.body_xy[:, 0], trial.body_xy[:, 1]
        at_cent = np.where((x > 460) & (x < 540) & (y > 450) & (y < 550))[0][0]

        C['s'].append(trial.body_speed[:at_cent])
        C['y'].append(y[:at_cent])
        C['x'].append(x[:at_cent])


# Normalize escape duration
L = {k:resample_list_of_arrayes_to_avg_len(v, N=100).mean(0) for k,v in L.items()}
R = {k:resample_list_of_arrayes_to_avg_len(v, N=100).mean(0) for k,v in R.items()}
C = {k:resample_list_of_arrayes_to_avg_len(v, N=100).mean(0) for k,v in C.items()}

f, ax = plt.subplots(figsize=(9, 9))
for arm, data, cmap in zip(arms, [L, C, R], cmaps):
    ax.scatter(data['x'][10:-2], data['y'][10:-2],
                    c=data['s'][10:-2], cmap=cmap,
                    s=200, lw=1, edgecolors='k',    )

ax.legend() 
ax.axis('off')
_ = ax.set(title='Mean trajectory per arm colored by speed')
save_figure(f, os.path.join(paths.plots_dir, 'M4 deaded end arm escape speed by arm with tracking'), svg=True)

# %%
"""
    Plot example trials for M4
"""
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


left_trials_idx = [i for i,t in trials.iterrows() if t.escape_arm == 'left'][0]
right_trials_idx = [i for i,t in trials.iterrows() if t.escape_arm == 'right'][7]
center_trials_idx = [i for i,t in trials.iterrows() if t.escape_arm == 'center'][0]
idxs = [left_trials_idx, center_trials_idx, right_trials_idx]

f, axarr = plt.subplots(figsize=(9*3, 9), ncols=3)
padding = 15
for arm, tidx, color, cmap, ax in zip(arms, idxs, colors, cmaps, axarr):
    trial = trials.loc[trials.index == tidx].iloc[0]
    # Get custom cmap
    norm=plt.Normalize(0, 15)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","#C24B91"])
    cmap.set_over("#C24B91")
    cmap.set_under("white")

    # Get when mouse is about to cross borders
    x, y = trial.body_xy[:, 0], trial.body_xy[:, 1]
    stop = np.where((x < 400 + padding) | (x > 600 - padding) | (y > 430 - padding))[0][0]

    # Plot
    plot_trial_tracking_as_lines(trial, ax, color, 4, color_by_speed=True,  
            stop_frame = stop,
            thick_lw=8, head_size=500, outline_width=3,
            cmap=cmap, outline_color=[.4, .4, .4], thin_alpha=0)



    ax.set(title=arm, xlim=[400, 600], ylim=[220, 450])

    # make colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='7%', pad=0.05)
    cb1 = mpl.colorbar.ColorbarBase(cax, 
                                cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label('Speed (a.u.)')
    ax.axis('off')
save_figure(f, os.path.join(paths.plots_dir, 'M4 deaded end arm escape speed by arm example'), svg=True)

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=5, vmax=10)

cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb1.set_label('Some Units')
fig.show()

# %%
"""
    Plot out-of-T time for each set of trials
"""

L, C, R = [], [], []
for i, trial in trials.iterrows():
    if trial.escape_arm == 'left':
        L.append(trial.time_out_of_t)
    elif trial.escape_arm == 'right':
        R.append(trial.time_out_of_t)
    else:
        C.append(trial.time_out_of_t)

Lstd, Cstd, Rstd = np.std(L), np.std(C), np.std(R)
L, C, R = np.mean(L), np.mean(C), np.mean(R)

f, ax = plt.subplots(figsize=(9, 9))
ax.errorbar([0, 1, 2], [L, C, R], yerr=[Lstd, Cstd, Rstd], lw=2, color=[.4, .4, .4])
ax.scatter([0, 1, 2], [L, C, R], c=colors, s=200, zorder=99, lw=1, edgecolors=[.4, .4, .4])

ax.set(ylabel='time (s)', xticks=[0, 1, 2], xticklabels=['left', 'center', 'right'], title='Time out of T')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'M4 deaded end time of out of T'), svg=True)

# %%
