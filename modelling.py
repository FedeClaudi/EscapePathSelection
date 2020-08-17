# %%
# Imports
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt
import statsmodels.api as sm
from tqdm import tqdm
from random import sample, choice
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split


from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes, get_euclidean_dist_for_dataset

# %%
# ----------------------------- Get mazes metadata ---------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = None,
    tracking = 'all'
)

trials = TrialsLoader(**params)
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials

print("Mazes stats")
_mazes = get_mazes()

euclidean_dists, eucl_dists_means, eucl_dists_traces = get_euclidean_dist_for_dataset(trials.datasets, trials.shelter_location)
eucls = {}
for m, data in eucl_dists_means.items():
    eucls[m] = {arm:np.mean(dt) for arm, dt in data.items()}

mazes = {k:m for k,m in _mazes.items() if k in paper.five_mazes}

mazes_stats = pd.DataFrame(dict(
    maze = list(mazes.keys()),
    geodesic_ratio = [v['ratio'] for v in mazes.values()],
    euclidean_ratio = list(euclidean_dists.values()),
    angles_ratio = [m['left_path_angle']/m['right_path_angle'] for m in mazes.values()]
))
mazes_stats
mazes = {m:data for m, data in _mazes.items() if m in paper.five_mazes}



# %%

alpha = .5  # weight of geodesic distance
beta = 1 # weight of euclidean distance
sigma = 150 # noise variance

niters = 1000
ntrials = 1000

f, axarr = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(14, 12))
axarr = axarr.flatten()

for ax, (maze, data) in zip(axarr, mazes.items()):
    lcol = paper.maze_colors[maze]
    rcol = desaturate_color(lcol)

    lengths = [data['left_path_length'], data['right_path_length']]

    # Get distributions of weighted lengths + noise
    distributions = {}
    for ln, col, side in zip(lengths, [lcol, rcol], ['left', 'right']):
        lens = []
        for i in np.arange(niters):
            lens.append(alpha * ln + beta * eucls[maze][side] + np.random.normal(0, sigma))

        ax.hist(lens, alpha=.4,  color=col, bins=25)
        ax.hist(lens, alpha=1, ec='k', histtype='step', bins=25)

        distributions[side] = lens

    
    # Get p(R) after N random trials
    outcomes = 0
    for trial in np.arange(ntrials):
        l = choice(distributions['left'])
        r = choice(distributions['right'])
        
        if r <= l: outcomes += 1


    ax.set(title=f'{maze} - p(R): {round(outcomes/ntrials, 3)}')




# %%
