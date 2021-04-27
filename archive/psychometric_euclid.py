# %%
# Imports
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt
import matplotlib.pyplot as plt

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.colors import *
from fcutils.plotting.plot_elements import plot_mean_and_error
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution
from fcutils.maths.filtering import smooth_hanning

import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes, get_euclidean_dist_for_dataset
from paper.utils.misc import resample_list_of_arrayes_to_avg_len


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

euclidean_dists, mean_eucl_dist, eucl_dist_traces = get_euclidean_dist_for_dataset(trials.datasets, trials.shelter_location)
mazes = {k:m for k,m in _mazes.items() if k in paper.five_mazes}

mazes_stats = pd.DataFrame(dict(
    maze = list(mazes.keys()),
    geodesic_ratio = [v['ratio'] for v in mazes.values()],
    euclidean_ratio = list(euclidean_dists.values(),)
))
mazes_stats


for maze, data in mean_eucl_dist.items():
    meanl = np.mean(data['left'])
    meanr = np.mean(data['right'])

    print(f'{maze} - mean eucl dist - {meanl} / {meanr}')

# TODO plot psychometric with better X axis ?? 

# %%
"""
    Plot an histogram with mean euclidean distance from shelter for left and right escape per arm
"""

f, axarr = plt.subplots(ncols=5, figsize=(24, 7), sharey=True)
f.suptitle('Histogram of mean eucl.dist. to shelt. per arm')

for ax, (maze, means) in zip(axarr, mean_eucl_dist.items()):
    color = paper.maze_colors[maze]    
    ax.hist(means['left'], bins=15, color=color, density=True, alpha=.4, label='left')
    ax.hist(means['right'], bins=15, color=desaturate_color(color), density=True, alpha=.4, label='right')

    ax.hist(means['left'], bins=15, histtype='step', color=color, density=True, alpha=1)
    ax.hist(means['right'], bins=15, histtype='step', color=desaturate_color(color), density=True, alpha=1)

    ax.legend()

    ax.set(title=maze, xlabel='mean eucl dist')

axarr[0].set( ylabel='density')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'mean eucl dist hist'))


# %%
"""
    Plot the average euclidean distance as a function of progression  during the escape

"""


f, axarr = plt.subplots(ncols=5, figsize=(24, 7), sharey=True)
f.suptitle('Avg eucl shelter dist during escape')

for ax, (maze, means) in zip(axarr, mean_eucl_dist.items()):
    color = paper.maze_colors[maze]    

    left = resample_list_of_arrayes_to_avg_len(eucl_dist_traces[maze]['left'], N=100)
    right = resample_list_of_arrayes_to_avg_len(eucl_dist_traces[maze]['right'], N=100)

    meanl = smooth_hanning(np.nanmean(left, 0), window_len=5)[6:-6]
    meanr = smooth_hanning(np.nanmean(right, 0), window_len=5)[6:-6]
    stdl = smooth_hanning(np.nanstd(left, 0), window_len=5)[6:-6]
    stdr = smooth_hanning(np.nanstd(right, 0), window_len=5)[6:-6]

    plot_mean_and_error(meanl, stdl, ax, color=color, label='left')
    plot_mean_and_error(meanr, stdr, ax, color=desaturate_color(color), label='right')
    ax.legend()

    ax.set(title=maze, xlabel='norm. escape progression')



axarr[0].set(ylabel='Euclidea dist to shelter')
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'mean eucl dist trace'), svg=True)

# %%

"""
    Plot overall mean of euclidean ratio
"""

f, ax = plt.subplots(figsize=(16, 9))


for n, (maze, means) in enumerate(mean_eucl_dist.items()):
    meanl = np.mean([np.mean(d) for d in eucl_dist_traces[maze]['left']])
    meanr = np.mean([np.mean(d) for d in eucl_dist_traces[maze]['right']])

    color = paper.maze_colors[maze]    


    ax.bar([n-.15, n+.15], [meanl, meanr], 
        color=[desaturate_color(color), color], width=.3)

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'real mean eucl dist trace'), svg=True)


# %%
