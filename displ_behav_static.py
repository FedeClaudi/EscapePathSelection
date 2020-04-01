"""
    Code to display the behaviour in a static manner. e.g. show tracking traces
"""


# %%
# Imports
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt
from matplotlib.patches import Ellipse

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic, get_distribution
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.colors import *
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as calc_angle

import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.dbase.TablesDefinitionsV4 import TrackingData


# %%
# --------------------------------- Load data -------------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = 1, 
    tracking = 'all'
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
# ---------------------------------------------------------------------------- #
#                                 PLOT TRACKING                                #
# ---------------------------------------------------------------------------- #
# -------------------------------- All trials -------------------------------- #
f, axarr = create_figure(subplots=True, ncols=3, nrows=2, figsize=(15, 10))

good_trials = {ds:dict(left=None, right=None) for ds in trials.datasets.keys()}

for ax, (ds, data) in zip(axarr, trials.datasets.items()):
    # Plot all trials
    for i, trial in data.iterrows():
        ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1], lw=.5, color=[.8, .8, .8], alpha=.7)

    # Select two random left vs right trials and plot
    for side, col in zip(['left', 'right'], [paper.maze_colors[ds], desaturate_color(paper.maze_colors[ds])]):
        good = False
        while not good:
            tr = data.loc[(data.escape_arm == side)].sample(1).iloc[0]
            if tr.body_xy[0, 1] <= 200: good = True
            good_trials[ds][side] = tr
        ax.plot(tr.body_xy[:, 0], tr.body_xy[:, 1], lw=4.5, color=col, alpha=1)

    ax.set(xticks=[], yticks=[], title=ds)

axarr[-1].axis("off")

clean_axes(f)




# %%
# ----------------------- Better look at example trials ---------------------- #

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





# %%

# %%
# %%
f, axarr = create_figure(ncols=10, nrows=8, figsize=(30, 20))

for ax, (i, tr) in zip(axarr, trials.datasets['maze6'].iterrows()):
    if tr.escape_arm == 'left':
        color = salmon
    else:
        print(np.min(tr.body_xy[:, 0]))
        color = green

    ax.plot(tr.body_xy[:, 0], tr.body_xy[:, 1], color=color)

    ax.set(xlim=[200, 800], ylim=[100, 900], yticks=[])



# %%
