"""
Analysis script for psychometric maze excluding alernative hypotheses:
        - Exploration
        - outward path
        - orientation
        - position
"""


# %%
# Imports
import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import seaborn as sns

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point, ball_and_errorbar
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.dbase.TablesDefinitionsV4 import *


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


# ---------------------------------- cleanup --------------------------------- #
"""
    In some trials on M1 mice go left first and then right, discard these trials.
"""
goodids, skipped = [], 0
_trials = trials.datasets['maze1']
for i, trial in _trials.iterrows():
    if trial.escape_arm == "left":
        if np.max(trial.body_xy[:, 0]) > 600:
            skipped += 1
            continue
    goodids.append(trial.stimulus_uid)

t = trials.datasets['maze1'].loc[trials.datasets['maze1'].stimulus_uid.isin(goodids)]
trials.datasets['maze1'] = t


# ------------------------------- Get mazes stats ------------------------------ #
_mazes = get_mazes()







# %%
# ---------------------------------------------------------------------------- #
#                           EFFECT OF ORIGIN ANALYSIS                          #
# ---------------------------------------------------------------------------- #

f, ax = plt.subplots(figsize=(16, 9))

# Do grouped bayes on trials within each dataset grouped by arm of origin
for dn, (ds, trs) in enumerate(trials.datasets.items()):
    l_origin = trs.loc[trs.origin_arm == 'left']
    r_origin = trs.loc[trs.origin_arm == 'right']

    dsets = dict(left=l_origin, right=r_origin)
    hits, ntrials, p_r, n_mice, trs = trials.get_binary_trials_per_dataset(dsets)

    results = {"dataset":[], "alpha":[], "beta":[], "mean":[],      
                "median":[], "sigmasquared":[], "prange":[],
                "distribution":[],}
    for (cond, h), n in zip(hits.items(), ntrials.values()):
        res = trials.grouped_bayes_analytical(np.sum(n), np.sum(h))
        results['dataset'].append(cond)
        results['alpha'].append(res[0])
        results['beta'].append(res[1])
        results['mean'].append(res[2])
        results['median'].append(res[3])
        results['sigmasquared'].append(res[4])
        results['prange'].append(res[5])
        results['distribution'].append(res[6])

    results = pd.DataFrame(results)

    ax.scatter(dn-.1, results['mean'][0], color=paper.maze_colors[ds], s=250)
    ax.scatter(dn+.1, results['mean'][1], color=desaturate_color(paper.maze_colors[ds]), s=250)
    ax.errorbar(dn-.1, results['mean'][0], sqrt(results['sigmasquared'][0]), color=paper.maze_colors[ds])
    ax.errorbar(dn+.1, results['mean'][1], sqrt(results['sigmasquared'][1]), color=desaturate_color(paper.maze_colors[ds]))
    ax.plot([dn-.1, dn+.1], [results['mean'][0], results['mean'][1]], lw=4, ls="--", color=paper.maze_colors[ds], zorder=0)


_ = ax.set(title="p(R) vs arm of origin", ylabel="p(R|origin)", xlabel="maze|arm of origin",
                xticks=[-.1, .1, .9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1], 
                xticklabels=['m1-L', 'R', 'm2-L', 'R', 'm3-L', 'R', 'm4-L', 'R', 'm6-L', 'R'],
                )
clean_axes(f)












# %%
# ---------------------------------------------------------------------------- #
#                             EXPLORATION ANALYSIS                             #
# ---------------------------------------------------------------------------- #
# ------------------------- Get sessions for datasets ------------------------ #
datasets_sessions = trials.get_datasets_sessions()
dataset_explorations = {}

for ds, sessions in datasets_sessions.items():
    dataset_explorations[ds] = []
    for sess in sessions['session_name']:
        dataset_explorations[ds].append((Explorations & f"session_name='{sess}'").fetch1())

dataset_explorations = {ds: pd.DataFrame(d) for ds, d in dataset_explorations.items()}

# %%
# ----------------------- Look at exploration duration ----------------------- #
f, axarr = plt.subplots(ncols = 2)


for n, (ds, data) in enumerate(dataset_explorations.items()):
    ball_and_errorbar(n, data.duration_s.mean(), data.duration_s.values, axarr[0], orientation='vertical',
                    s=250, color=paper.maze_colors[ds])
    
    ball_and_errorbar(n, data.distance_covered.mean(), data.distance_covered.values, axarr[1], 
                    orientation='vertical',
                    s=250, color=paper.maze_colors[ds])

axarr[0].set(title="mean exploration duration", ylabel="duration (s)", xlabel="maze",
                xticks=[0, 1, 2, 3, 4], xticklabels=paper.five_mazes)
axarr[1].set(title="mean exploration distance covered", ylabel="distance (px)", xlabel="maze",
                xticks=[0, 1, 2, 3, 4], xticklabels=paper.five_mazes)
clean_axes(f)



# %%
# ----------------- Look at ammount of time spend on each arm ---------------- #
NORM_BY_MAZE_ASYM = True


res = {ds:{'left':[], 'right':[]} for ds in dataset_explorations.keys()}
for n, (ds, data) in enumerate(dataset_explorations.items()):
    for i, row in data.iterrows():
        trk = row.body_tracking

        l = len(trk[trk[:, 0] < 450])/trk.shape[0]
        r = len(trk[trk[:, 0] > 550])/trk.shape[0]

        if NORM_BY_MAZE_ASYM:
            l = l / _mazes[ds]['left_path_length'] * _mazes[ds]['right_path_length']

        res[ds]['left'].append(l)
        res[ds]['right'].append(r)
res = {ds: pd.DataFrame(d) for ds, d in res.items()}



# Plot avg occupancy normalised for exploration of each mouse
f, ax = plt.subplots(ncols = 1)
for n, (ds, data) in enumerate(res.items()):
    ball_and_errorbar(n -.1, data.left.mean(), data.left.values, 
                    ax, orientation='vertical',
                    s=250, color=paper.maze_colors[ds])
    ball_and_errorbar(n + .1, data.right.mean(), data.right.values, 
                    ax, orientation='vertical',
                    s=250, color=desaturate_color(paper.maze_colors[ds]))

    ax.plot([n -.1, n+.1], [data.left.mean(), data.right.mean()], 
                    lw = 3, ls="--", color=paper.maze_colors[ds])

ax.set(title="Arm occupancy in exploration", ylabel="Occupancy", xlabel="maze",
                xticks=[-.1, .1, .9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1], 
                xticklabels=['m1-L', 'R', 'm2-L', 'R', 'm3-L', 'R', 'm4-L', 'R', 'm6-L', 'R'],
                )

if NORM_BY_MAZE_ASYM:
    ax.set(ylabel="Length normalised occupancy")

clean_axes(f)


# %%
# -------------------------- Heatmap of explorations ------------------------- #
res = {ds:None for ds in dataset_explorations.keys()}
for n, (ds, data) in enumerate(dataset_explorations.items()):
    ds_trk = []
    for i, row in data.iterrows():
        ds_trk.append(row.body_tracking)
    res[ds] = np.vstack(ds_trk)

# for ds, data in res.items():
#     data[(data[:, 0] > 450) & (data[:, 0] < 550)] = np.nan

f, axarr = create_figure(subplots=True, ncols=3, nrows=2, figsize=(15, 10))

for ax, (ds, data) in zip(axarr, res.items()):

    ax.hexbin(data[:, 0], data[:, 1], mincnt=30, gridsize=100, bins='log')



