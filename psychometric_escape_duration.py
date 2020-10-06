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
from paper.helpers.mazes_stats import get_mazes
from paper.utils.misc import run_multi_t_test_bonferroni, run_multi_t_test_bonferroni_one_samp_per_item

# %%
_mazes = get_mazes()

# %%
# --------------------------------- Load data -------------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'all',
    escapes_dur = True,
)

trials = TrialsLoader(**params)
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials
trials.keep_catwalk_only()

# %%


f, axarr = plt.subplots(nrows=2, figsize=(16, 9), sharex=True)
f2, ax  = plt.subplots(figsize=(8, 10))

meandurs = {}

for n, (maze, trs) in enumerate(trials.datasets.items()):
    # Get mean duration per arm
    left = trs.loc[trs.escape_arm == 'left'].escape_duration.mean()
    right = trs.loc[trs.escape_arm == 'right'].escape_duration.mean()

    # Get all durations per arm
    meandurs[maze] = {'l':trs.loc[trs.escape_arm == 'left'].escape_duration, 
                        'r':trs.loc[trs.escape_arm == 'right'].escape_duration}

    # Get std of duration per arm
    lstd = trs.loc[trs.escape_arm == 'left'].escape_duration.std()
    rstd = trs.loc[trs.escape_arm == 'right'].escape_duration.std()


    # Get low percentile of duration per arm
    llowperc = np.percentile(trs.loc[trs.escape_arm == 'left'].escape_duration, 10)
    rlowperc = np.percentile(trs.loc[trs.escape_arm == 'right'].escape_duration, 10)

    # Plot ratios of low percentiles in separate figure
    ax.bar(n, llowperc/rlowperc, color=paper.maze_colors[maze])

    # Plot left vs right mean duration + std
    axarr[0].errorbar([n-.15, n+.15], [left, right], yerr=[lstd, rstd],  color=paper.maze_colors[maze],
                    ms=16, lw=6, elinewidth =2)
    axarr[0].plot([n-.15, n+.15], [left, right], 'o-',  color=paper.maze_colors[maze],
                    lw=6, ms=16)
    axarr[0].scatter([n-.15, n+.15], [llowperc, rlowperc], color=paper.maze_colors[maze], zorder=99, ec='k')
    axarr[1].plot([n-.15, n+.15], [left/right, right/right], 'o-', color=paper.maze_colors[maze],
                    ms=16, lw=6)


# Run ptest and add to figure
significant, pval = run_multi_t_test_bonferroni(meandurs)
for n, sig in enumerate(significant):
    if sig:
        axarr[0].text(n, 7, '*', fontsize=25, fontweight=500, horizontalalignment='center')


# Cleanup axes
axarr[0].set(title='Escape duration by path', ylabel='mean duration (s)', xticks=[0, 1, 2, 3, 4,],)
axarr[1].set(ylabel='norm. duration', xticklabels=trials.datasets.keys(),  xticks=[0, 1, 2, 3, 4,],)
axarr[1].axhline(1, ls='--', zorder=-1, lw=2, color=[.6, .6, .6])


clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f"escape duration by path"), svg=True)

ax.set(title='Fast escape ratio per maze', ylabel='L/R', xticklabels=trials.datasets.keys(),  xticks=[0, 1, 2, 3, 4,],)
clean_axes(f2)
save_figure(f2, os.path.join(paths.plots_dir, f"lowperc_escape duration ratio_by_arm"), svg=True)


# %%
"""
    Look at distribution of out-of-T times. 
"""
def get_lr_stats(data):
    left = data.loc[data.escape_arm == 'left'].time_out_of_t.mean()
    right = data.loc[data.escape_arm == 'right'].time_out_of_t.mean()

    lstd = data.loc[data.escape_arm == 'left'].time_out_of_t.std()
    rstd = data.loc[data.escape_arm == 'right'].time_out_of_t.std()

    return left, right, lstd, rstd

f, ax  = plt.subplots(figsize=(16, 9))


meandata = {}
for n, (maze, trs) in enumerate(trials.datasets.items()):
    left, right, lstd, rstd = get_lr_stats(trs)

    meandata[maze] = trs.time_out_of_t.values

    ax.errorbar([n-.15, n+.15], [left, right], yerr=[lstd, rstd],  color=paper.maze_colors[maze],
                    ms=16, lw=6, elinewidth =2)
    ax.plot([n-.15, n+.15], [left, right], 'o-',  color=paper.maze_colors[maze],
                    lw=6, ms=16)

    # Look at individual mice:
    for mouse in trs.mouse_id.unique():
        mouse_trials = trs.loc[trs.mouse_id == mouse]
        left, right, lstd, rstd = get_lr_stats(mouse_trials)

        ax.scatter([n-.20 + np.random.normal(0, .02), n+.20 + np.random.normal(0, .02)], 
                    [left, right], zorder=-1, ec='k', lw=1, s=40,
                    color=desaturate_color(paper.maze_colors[maze]), alpha=.8)


# Run multi test bonferroni
sig, p, pairs = run_multi_t_test_bonferroni_one_samp_per_item(meandata)
x_offsets = dict(
    maze1 = 0,
    maze2 = 1,
    maze3 = 2, 
    maze4 = 3,
    maze6 = 4
)

yoff = 5
for issig, (m1, m2) in zip(sig, pairs):
    if issig:
        print(m1, m2)
        ax.errorbar([x_offsets[m1], x_offsets[m2]],[yoff, yoff], yerr=.1, lw=2, color='k')
        yoff += .2


_ = ax.set(title='Time out of threat platform by maze and arm', ylabel='mean duration (s)', 
            xticks=[0, 1, 2, 3, 4,], xticklabels=[maze for maze in trials.datasets.keys()])

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f"time out of T by maze and arm"))

# %%
"""
    Same, but without spitting between left and right

"""
def plot(mn, std, color, ms, off):
    ax.errorbar([n + off], [mn], yerr=[std],  color=color,
                ms=ms, lw=6, elinewidth =2)
    ax.plot([n + off], [mn], 'o-',  color=color,
                    lw=2, ms=ms)


f, ax  = plt.subplots(figsize=(16, 9))
for n, (maze, trs) in enumerate(trials.datasets.items()):
    mn, std = trs.time_out_of_t.mean(), trs.time_out_of_t.std()

    plot(mn, std, paper.maze_colors[maze], 16, 0)

    for mouse in trs.mouse_id.unique():
        mouse_trials = trs.loc[trs.mouse_id == mouse]
        mn, std = mouse_trials.time_out_of_t.mean(), mouse_trials.time_out_of_t.std()

        ax.plot([n + .1 + np.random.normal(0, .02)], [mn], 'o-',  color=[.7, .7, .7],
                    lw=2, ms=8, zorder=-1)


yoff = 5
for issig, (m1, m2) in zip(sig, pairs):
    if issig:
        print(m1, m2)
        ax.errorbar([x_offsets[m1], x_offsets[m2]],[yoff, yoff], yerr=.1, lw=2, color='k')
        yoff += .2


_ = ax.set(title='Time out of threat platform by maze and arm', ylabel='mean duration (s)', 
            xticks=[0, 1, 2, 3, 4,], xticklabels=[maze for maze in trials.datasets.keys()])

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f"time out of T by maze"))

# %%

# %%

# %%
