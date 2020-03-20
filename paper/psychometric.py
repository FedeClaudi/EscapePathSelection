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
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes



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
#                                 Psychometric                                 #
# ---------------------------------------------------------------------------- #

# ? Options
PLOT_INDIVIDUALS = True
ADD_M6 = True

# ------------------------------- Prepare Data ------------------------------- #
if not ADD_M6:
    mazes = {k:m for k,m in _mazes.items() if k in paper.psychometric_mazes}

    X_labels = list(trials.datasets.keys())[:-1]
    X = [maze['ratio'] for maze in mazes.values()]
    Y = grouped_pRs['mean'].values[:-1]

    Y_err = [sqrt(v)*2 for v in grouped_pRs['sigmasquared']][:-1]
    colors = [paper.maze_colors[m] for m in X_labels]

    xfit = X
    yfit = Y
    yerror = Y_err

    hierarchical = hierarchical_pRs.loc[hierarchical_pRs.dataset != 'Maze6']

else:
    mazes = {k:m for k,m in _mazes.items() if k in paper.five_mazes}
    X_labels = list(trials.datasets.keys())
    X = np.array([maze['ratio'] for maze in mazes.values()])
    Y = grouped_pRs['mean'].values
    Y_err = [sqrt(v)*2 for v in grouped_pRs['sigmasquared']]
    colors = [paper.maze_colors[m] for m in X_labels]

    xfit = X[:-1]
    yfit = Y[:-1]
    yerrfit = Y_err[:-1]

    hierarchical = hierarchical_pRs


# Colors
xmin, xmax = -1, 3

# Create figure
f, ax = create_figure(subplots=False, figsize=(16, 10))


# ---------------------------- Plot means scatter ---------------------------- #
ax.scatter(X, Y, c=colors, s=250, ec='k', zorder=99)

for x,y,yerr,color in zip(X, Y, Y_err, colors):
    _ = ax.errorbar(x, y, yerr=yerr, fmt = 'o', c=color, lw=4)
    _ = hline_to_point(ax, x, y, color=color, ls="--", alpha=.3, lw=3, xmin=xmin-3)
    _ = vline_to_point(ax, x, y, color=color, ls="--", alpha=.3, lw=3, ymin=-1)


# ------------------------ Plot indidividuals scatter ------------------------ #
if PLOT_INDIVIDUALS:
    for x, dset, color in zip(X, X_labels, colors):
        ys = hierarchical.loc[hierarchical.dataset == dset]['means'].values[0]
        xs = np.random.normal(x, .01, size=len(ys))
        color = desaturate_color(color)

        ax.scatter(xs, ys, color=color, s=50, ec=[.2, .2, .2], alpha=.5, zorder=98)

# ------------------------------ Fit/Plot curve ------------------------------ #

curve_params = plot_fitted_curve(centered_logistic, xfit, yfit, 
                ax, xrange=[xmin, xmax], 
                scatter_kwargs=dict(alpha=0),
                fit_kwargs = dict(sigma=yerrfit),
                line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))


_ = ax.set(title="p(R) psychometric",
        xlim = [xmin, xmax],
        ylim = [0, 1],
        xticks = X,
        xticklabels = X_labels,
        xlabel='Path length asymmetry',
        ylabel='p(R)'
        )
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'psychometric_M6_{ADD_M6}_individuals_{PLOT_INDIVIDUALS}'), svg=True)




# %%
# ----------------------------- Maze 4 vs maze 6 ----------------------------- #
f, axarr = create_figure(subplots=True, ncols=2, figsize=(16, 10))

X_labels =['Maze4', 'Maze6']
A = grouped_pRs.loc[grouped_pRs.dataset.isin(['maze4', 'maze6'])]['alpha'].values
B = grouped_pRs.loc[grouped_pRs.dataset.isin(['maze4', 'maze6'])]['beta'].values
_colors = colors[-2:]

# Plot posteriors for the two mazes
dists = []
for a, b, label, color in zip(A, B, X_labels, _colors):
    beta = get_distribution('beta', a, b, n_samples=80000)
    dists.append(beta)
    plot_kde(ax=axarr[0], data=beta, vertical=True, z=0, 
                color=color, label=label)

axarr[0].legend()
axarr[0].axhline(0.5, ls=':', color=[.6, .6, .6])
_ = axarr[0].set(title='Maze 4 vs Maze 6 | p(R)', 
        xlabel='density', 
        ylabel='p(R)', 
        ylim=[0, 1], 
    )


# Plot delta of the posteriors
delta = dists[0] - dists[1]
plot_kde(ax=axarr[1], data=delta, vertical=True, z=0, 
            color=[.4, .4, .4], label='M4 - M6')
axarr[1].axhline(0, ls=':', color=[.6, .6, .6])
_ = axarr[1].set(title='Maze 4 - Maze 6 | p(R)', 
        xlabel='delta p(R) [M4 - M6]', 
        ylabel='p(R)', 
        ylim=[-.5, .5], 
    )


clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'm4_m6_pR'), svg=True)


# %%
# --------------------------- Print n trials summary -------------------------- #
summary = pd.DataFrame(dict(
    maze=X_labels,
    tot_trials = [len(trials.datasets[m]) for m in X_labels],
    n_mice = list(n_mice.values()),
    avg_n_trials_per_mouse = [np.mean(nt) for nt in list(ntrials.values())]
))
summary



# %%