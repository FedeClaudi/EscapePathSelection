"""
Time analysis for psychometric mazes

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
from fcutils.maths.distributions import get_distribution
from fcutils.plotting.colors import desaturate_color


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes


# %%
# --------------------------------- Load data -------------------------------- #

# ? laoding data is the same as for psychometric.py

print("Loading data")
params = dict(
    naive = None,
    lights = 1, 
    tracking = 'all'
)

trials = TrialsLoader(**params)
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials


# ------------------------------ Get first trial ----------------------------- #
first_trials={k:v.loc[v.trial_num_in_session == 0] for k,v in trials.datasets.items()}
# first_trials_pr = {k:len(v.loc[v.escape_arm == 'right'])/len(v) for k,v in first_trials.items()}
grouped_first_trial_pr = trials.grouped_bayes_by_dataset_analytical(datasets=first_trials)



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
mazes = get_mazes()
mazes = {k:m for k,m in mazes.items() if k in paper.psychometric_mazes}


# %%
# ----------------------- Plot p(R) first trial vs all ----------------------- #
# Prep data
X_labels = list(first_trials.keys())
X = np.arange(len(first_trials.keys()))/2

Y_all = grouped_pRs['mean'].values
Y_err_all = [sqrt(v)*2 for v in grouped_pRs['sigmasquared']]

Y_first = grouped_first_trial_pr['mean'].values
Y_err_first = [sqrt(v)*2 for v in grouped_first_trial_pr['sigmasquared']]

colors = [paper.maze_colors[m] for m in X_labels]



f, ax = create_figure(subplots=False, figsize=(16, 10))

# Plot all trials bar plot
ax.bar(X, Y_all, width=.3, linewidth=0, yerr=Y_err_all, color=[desaturate_color(col) for col in colors],
            error_kw={'elinewidth':3})

# Plot the first trials as scatter
ax.scatter(X+.05, Y_first, c=colors, s=800, lw=2, ec='k', zorder=99)

for x, y, yerr, color in zip(X, Y_first, Y_err_first, colors):
    _ = ax.errorbar(x+.05, y, yerr=yerr, fmt = 'o', c=[.2, .2, .2], lw=3)

ax.set(title="p(R) on first trial vs global p(R)", 
            xticks=X, xticklabels=X_labels,
            xlabel='Maze', ylabel='p(R)')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'first_trial_pR'), svg=True)





# %%
# Plot p(R) as a function of time
start_time = 0
end_time = 3000
window_size = 240
start_times = np.arange(start_time, end_time, window_size/2)

f, axarr = create_figure(subplots=True, nrows=len(X_labels) , sharex=True, sharey=True,
                    figsize=(16, 20))

for ax, label, color in zip(axarr, X_labels, colors):
    trs = trials.datasets[label]

    # Plot vertical KDEs with posteriros of bayes on binned trials
    for window_n, start_time in enumerate(start_times):
        trs_in_window = trs.loc[(trs.stim_time_s >= start_time) & (trs.stim_time_s < start_time+window_size)]

        n_trials = len(trs_in_window)
        n_right = len(trs_in_window.loc[trs_in_window.escape_arm == 'right'])

        if n_trials > 3:
            a, b, mean, _, sigmasquared, _, beta = trials.grouped_bayes_analytical(n_trials, n_right)
            if b == 0: b += .5
            beta = get_distribution('beta', a, b, n_samples=5000)

            plot_kde(ax=ax, data=beta, vertical=True, z=start_time+window_size*.5, 
                        normto=window_size * .75, color=desaturate_color(color))
            

    # Plot a vertical line with global p(R)
    ax.axhline(grouped_pRs.loc[grouped_pRs.dataset == label]['mean'].values[0],  
                    color=color, lw=3, ls="--")

axarr[0].set(title='Maze1 | Timed p(R)', ylabel='p(R)', ylim=[0, 1])
axarr[1].set(title='Maze2', ylabel='p(R)')
axarr[2].set(title='Maze3', ylabel='p(R)')
axarr[3].set(title='Maze3', ylabel='p(R)', xlabel='Time (s)')


clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'timed_pR'), svg=True)




# %%
# ---------------------------------------------------------------------------- #
#                              NAVIE VS NOT NAIVE                              #
# ---------------------------------------------------------------------------- #

params = dict(
    naive = 0,
    lights = 1, 
    tracking = 'all'
)

# --------------------------------- Load data -------------------------------- #
trials = TrialsLoader(**params)
trials.datasets['maze1_not_naive'] = trials.load_trials_by_condition(maze_design=1)
trials.naive=1
trials.datasets['maze1_naive'] = trials.load_trials_by_condition(maze_design=1)

# -------------------------------- Clean data -------------------------------- #
for key in ['maze1_not_naive', 'maze1_naive']:
    goodids, skipped = [], 0
    _trials = trials.datasets[key]
    for i, trial in _trials.iterrows():
        if trial.escape_arm == "left":
            if np.max(trial.body_xy[:, 0]) > 600:
                skipped += 1
                continue
        goodids.append(trial.stimulus_uid)

    t = trials.datasets[key].loc[trials.datasets[key].stimulus_uid.isin(goodids)]
    trials.datasets[key] = t

# --------------------------------- Get p(R) --------------------------------- #
grouped_pRs = trials.grouped_bayes_by_dataset_analytical()


# %%
# Plot naive vs not naive posteriors

f, ax = create_figure(subplots=False, figsize=(16, 10))


X = [0, .25]
X_labels =['Not naive', 'naive']
Y = grouped_pRs['mean'].values
Y_err = [sqrt(v)*2 for v in grouped_pRs['sigmasquared']]

_colors = [colors[0], desaturate_color(colors[0], k=.4)]

ax.scatter(X, Y, c=_colors, s=500, ec='k', zorder=99)
ax.plot(X, Y, lw=5, color=[.4, .4, .4], ls='--', alpha=.6)

for x,y,yerr,color in zip(X, Y, Y_err, _colors):
    _ = ax.errorbar(x, y, yerr=yerr, fmt = 'o', c=desaturate_color(color, k=.2), lw=4)


_ = ax.set(title='Naive vs experience p(R) | Maze 1', xlabel='Maze', ylabel='p(R)', 
        xticks=X, xticklabels=X_labels, ylim=[0, 1], xlim=[-.05, .3])

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'naive_vs_experienced_pR'), svg=True)
