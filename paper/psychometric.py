"""
Analysis script for psychometric plot
"""


# %%
# Imports
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.plot_distributions import plot_fitted_curve
from fcutils.maths.distributions import centered_logistic


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes



# %%
# --------------------------------- Load data -------------------------------- #
params = dict(
    naive = None,
    lights = None, 
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


# --------------------------- P(R) and other stuff --------------------------- #
hits, ntrials, p_r, n_mice, trs = trials.get_binary_trials_per_dataset()
grouped_pRs = trials.bayes_by_dataset_analytical()

# ----------------------------- Get mazes metadata ---------------------------- #
mazes = get_mazes()
mazes = {k:m for k,m in mazes.items() if k in ['m1', 'm2', 'm3', 'm4']}




# %%

# ---------------------------------------------------------------------------- #
#                                 Psychometric                                 #
# ---------------------------------------------------------------------------- #

# ------------------------------- Prepare Data ------------------------------- #
X_labels = mazes.keys()
X = [maze['ratio'] for maze in mazes.values()]
Y = grouped_pRs['mean'].values

# Y error is used for fitting
pranges = grouped_pRs['prange']
# Y_err = np.array([[y - prange.low, prange.high - y] for y,prange in zip(Y, pranges)])
Y_err = [sqrt(v) for v in grouped_pRs['sigmasquared']]

# Colors
colors = [paper.maze_colors[m] for m in X_labels]
xmin, xmax = -1, 3

# Create figure
f, ax = create_figure(subplots=False, figsize=(16, 10))


# ---------------------------- Plot means scatter ---------------------------- #
ax.scatter(X, Y, c=colors, s=250, ec='k', zorder=99)

for x,y,yerr,color in zip(X, Y, Y_err, colors):
    _ = ax.errorbar(x, y, yerr=yerr, fmt = 'o', c=color, lw=4)
    _ = hline_to_point(ax, x, y, color=color, ls="--", alpha=.3, lw=3, xmin=xmin-3)
    _ = vline_to_point(ax, x, y, color=color, ls="--", alpha=.3, lw=3, ymin=-1)

# ------------------------------ Fit/Plot curve ------------------------------ #
curve_params = plot_fitted_curve(centered_logistic, X, Y, ax, xrange=[xmin, xmax], 
                scatter_kwargs=dict(alpha=0),
                fit_kwargs = dict(sigma=Y_err),
                line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))


_ = ax.set(title="p(R) psychometric",
        xlim = [xmin, xmax],
        ylim = [0, 1],
        xticks = X,
        xticklabels = X_labels,
        )
clean_axes(f)



save_figure(f, os.path.join(paths.plots_dir, 'psychometric'), svg=True)






# %%
