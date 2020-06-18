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


# %%
f, axarr = plt.subplots(nrows=2, figsize=(16, 9), sharex=True)
f2, ax  = plt.subplots(figsize=(8, 10))

for n, (maze, trs) in enumerate(trials.datasets.items()):
    left = trs.loc[trs.escape_arm == 'left'].escape_duration.mean()
    right = trs.loc[trs.escape_arm == 'right'].escape_duration.mean()

    lstd = trs.loc[trs.escape_arm == 'left'].escape_duration.std()
    rstd = trs.loc[trs.escape_arm == 'right'].escape_duration.std()

    llowperc = np.percentile(trs.loc[trs.escape_arm == 'left'].escape_duration, 10)
    rlowperc = np.percentile(trs.loc[trs.escape_arm == 'right'].escape_duration, 10)

    ax.bar(n, llowperc/rlowperc, color=paper.maze_colors[maze])


    axarr[0].errorbar([n-.15, n+.15], [left, right], yerr=[lstd, rstd],  color=paper.maze_colors[maze],
                    ms=16, lw=6, elinewidth =2)
    axarr[0].plot([n-.15, n+.15], [left, right], 'o-',  color=paper.maze_colors[maze],
                    lw=6, ms=16)
    axarr[0].scatter([n-.15, n+.15], [llowperc, rlowperc], color=paper.maze_colors[maze], zorder=99, ec='k')
    axarr[1].plot([n-.15, n+.15], [left/right, right/right], 'o-', color=paper.maze_colors[maze],
                    ms=16, lw=6)


    # ratio = _mazes[maze]['left_path_length'] / _mazes[maze]['right_path_length']
    # axarr[2].plot([n-.15, n+.15], [(left/right)/ratio, right/right], 'o-')

axarr[0].set(title='Escape duration by path', ylabel='mean duration (s)', xticks=[0, 1, 2, 3, 4,],)
axarr[1].set(ylabel='norm. duration', xticklabels=trials.datasets.keys(),  xticks=[0, 1, 2, 3, 4,],)
axarr[1].axhline(1, ls='--', zorder=-1, lw=2, color=[.6, .6, .6])



clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f"escape duration by path"))

ax.set(title='Fast escape ratio per maze', ylabel='L/R', xticklabels=trials.datasets.keys(),  xticks=[0, 1, 2, 3, 4,],)
clean_axes(f2)
save_figure(f2, os.path.join(paths.plots_dir, f"lowperc_escape duration ratio_by_arm"))


# %%