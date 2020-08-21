# %%
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
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.colors import *
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as calc_angle

import paper
from paper import paths
from paper.trials import TrialsLoader


# ----------------------------------- Setup ---------------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'all',
    escapes_dur=True,
)

trials = TrialsLoader(**params)
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials

# %%
# ------------------------------- Trajectories ------------------------------- #
"""
    Plot all trials trajectories color coded by arm + the mean trajectory with error
"""
# TODO for M4 add trials with the central arm


# ! check that I'm only taking sessions that actually have the central arm in the maze
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


tloader = TrialsLoader(**params)
tloader.max_duration_th = 15
trials =  tloader.load_trials_by_condition(maze_design=4)
trials = trials.loc[trials.session_name.isin(keep_session_names)]

# TODO use these trials for plotting here but the other trials variable for plotting down

f, axarr = create_figure(subplots=True, ncols=5, figsize=(22, 7), sharex=True, sharey=True)

good_trials = {ds:dict(left=None, right=None) for ds in trials.datasets.keys()}

for ax, (ds, data) in zip(axarr, trials.datasets.items()):
    # data = data.loc[data.escape_arm != 'center']

    # Store left and right tracking for left and right trials
    lxs, lys = [], []
    rxs, rys = [], []


    # Plot all trials
    for i, trial in data.iterrows():
        ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1], lw=.5, color=[.6, .6, .6], alpha=.7)

        if trial.escape_arm == 'left':
            lxs.append(trial.body_xy[:, 0])
            lys.append(trial.body_xy[:, 1])
        else:
            rxs.append(trial.body_xy[:, 0])
            rys.append(trial.body_xy[:, 1])

    # Plot the average left and right trial
    # First resample


    mean_dur = np.nanmean([np.nanmean([len(i) for i in l]) for l in [lxs, lys, rxs, rys]]).astype(np.int32)
    lxs = np.nanmedian(np.vstack([resample(pd.Series(X).interpolate(), mean_dur) for X in lxs]), 0)
    lys = np.nanmedian(np.vstack([resample(pd.Series(X).interpolate(), mean_dur) for X in lys]), 0)
    rxs = np.nanmedian(np.vstack([resample(pd.Series(X).interpolate(), mean_dur) for X in rxs]), 0)
    rys = np.nanmedian(np.vstack([resample(pd.Series(X).interpolate(), mean_dur) for X in rys]), 0)

    col = paper.maze_colors[ds]

    for x, y, c in zip([lxs, rxs], [lys, rys], [col, desaturate_color(col)]):    
        ax.plot(x[5:-5], y[5:-5], lw=7, color=[1, 1, 1], alpha=1)
        ax.plot(x[5:-5], y[5:-5], lw=6, color=col, alpha=1)

    ax.axis('off')
    ax.set(xticks=[], yticks=[], title=ds)


clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'tracking_traces'), svg=True)




# %%
# ------------------------- Example trials trajectory ------------------------ #
"""
    For each maze plot the trajectory of example trials, one per arm, with
    the stick-figure mouse schematic. 
"""


bps = ['snout', 'neck', 'body', 'tail_base']

head_width = 30
body_width = 50
width_factor = .2

f, axarr = create_figure(subplots=True, ncols=3, nrows=2, figsize=(15, 10))


for ax, (maze, trs) in zip(axarr, good_trials.items()):
    for side, tr in trs.items():
        if side == 'right':
            xshift = 80
        else:
            xshift = 0
        # Get trial tracking for all bparts
        tracking = pd.DataFrame((TrackingData * TrackingData.BodyPartData & f"recording_uid='{tr.recording_uid}'").fetch())
        tracking = {bp:tracking.loc[tracking.bpname == bp].tracking_data.values[0] for bp in bps}
        tracking = {bp:trk[tr.stim_frame:tr.at_shelter_frame, :] for bp, trk in tracking.items()}

        # Plot position at frames
        every = 20
        for frame in np.arange(0, len(tracking['body']), 1):
            for bp1, bp2 in [['tail_base', 'body'], ['body', 'neck'], ['neck', 'snout']]:

                if bp1 != 'neck': 
                    color = [.4, .4, .4]
                    lw = 3
                    zorder = 90
                else: 
                    lw = 3
                    zorder=99
                    if side == 'left': color = paper.maze_colors[maze]
                    else: color = desaturate_color(paper.maze_colors[maze])

                if not frame % every == 0:
                    alpha=.075
                    zorder -= 20
                    lw = 3
                else:
                    alpha=1
                    lw=4

                # Plot a first time larger in white to get a better look
                # ax.plot(
                #     [tracking[bp1][frame, 0]+xshift, tracking[bp2][frame, 0]+xshift],
                #     [tracking[bp1][frame, 1], tracking[bp2][frame, 1]],
                #     lw=lw+1, color=[1, 1, 1], alpha=alpha, zorder=zorder,
                #     solid_joinstyle='round', solid_capstyle='round'
                # )
                ax.plot(
                    [tracking[bp1][frame, 0]+xshift, tracking[bp2][frame, 0]+xshift],
                    [tracking[bp1][frame, 1], tracking[bp2][frame, 1]],
                    lw=lw, color=color, alpha=alpha, zorder=zorder,
                    solid_joinstyle='round', solid_capstyle='round'
                )
    ax.set(xticks=[], yticks=[], title=maze)

    break

axarr[-1].axis("off")

clean_axes(f)
