"""
Analysis script for psychometric mazes -> GLM model
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
import statsmodels.api as sm

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

euclidean_dists = get_euclidean_dist_for_dataset(trials.datasets, trials.shelter_location)
mazes = {k:m for k,m in _mazes.items() if k in paper.five_mazes}

mazes_stats = pd.DataFrame(dict(
    maze = list(mazes.keys()),
    geodesic_ratio = [v['ratio'] for v in mazes.values()],
    euclidean_ratio = list(euclidean_dists.values(),)
))
mazes_stats



# %%

# ---------------------------------------------------------------------------- #
#                                 GLM MODELLING                                #
# ---------------------------------------------------------------------------- #

# ------------------------------ Fit on all DATA ----------------------------- #

# Get trials summary statistics per maze
summary = dict(maze = [],
                geodesic_ratio=[], 
                euclidean_ratio=[], 
                n=[],  # number of trials
                k=[],  # numebr of hits
                m=[],    # n - k
                )

for maze in hits.keys():
    summary['maze'].append(maze)
    summary['geodesic_ratio'].append(mazes_stats.loc[mazes_stats.maze == maze].geodesic_ratio.values[0])
    summary['euclidean_ratio'].append(mazes_stats.loc[mazes_stats.maze == maze].euclidean_ratio.values[0])
    summary['n'].append(np.sum(ntrials[maze]))
    summary['k'].append(np.sum(hits[maze]))
    summary['m'].append(summary['n'][-1] - summary['k'][-1])
summary= pd.DataFrame(summary)

# Fit model
exog = summary[['geodesic_ratio', 'euclidean_ratio']]
exog = sm.add_constant(exog, prepend=False)
endog = summary[['k', 'm']]

glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
res = glm_binom.fit()

# Predict mazes p(R) from model
nobs = res.nobs
y = endog['k']/endog.sum(1)
yhat = res.mu

# %%
# -------------------------- Plot as quality of fit -------------------------- #

f, ax = create_figure(subplots=False)

ax.scatter(grouped_pRs['mean'], yhat, c=[paper.maze_colors[m] for m in summary.maze.values],
                s=300)
ax.plot([0, 1], [0, 1], ls="--", lw=2, color='k', alpha=.4, zorder=-1)

ax.set(title="GLM quality of FIT", xlabel="real p(R)", ylabel="predicted p(R)")
clean_axes(f)

# train mulitiple times wit hold out one maze at the time
# ? Commented out because it doesn't make much of a difference
# predicts = {m:[] for m in summary.maze.values}
# for i in range(100):
#     holdout = np.random.randint(0, len(ex))
#     kept = [m for n,m in enumerate(predicts.keys()) if n != holdout]

#     ex = exog.copy().drop(index=holdout)
#     en = endog.copy().drop(index=holdout)

#     glm_binom = sm.GLM(en, ex, family=sm.families.Binomial())
#     res = glm_binom.fit()

#     # Predict mazes p(R) from model
#     nobs = res.nobs
#     y = endog['k']/endog.sum(1)
#     pred = res.mu

#     for p,m in zip(pred, kept): predicts[m].append(p)


# for maze, preds in predicts.items():
#     ax.errorbar(grouped_pRs.loc[grouped_pRs.dataset == maze]['mean'].values[0], np.mean(preds), 
#                     yerr=np.std(preds))

# %%
# ----------------------------- Plot as a heatmap ---------------------------- #
f, ax = create_figure(subplots=False)

ax.scatter(mazes_stats.geodesic_ratio, mazes_stats.euclidean_ratio, 
                    c=grouped_pRs['mean'],
                    ec='k',
                    lw=2,
                    cmap='bwr', 
                    vmin=0, vmax=1, 
                    s=400)

# Plot a line over where pR = 0.5
x0 = np.linspace(0, 1, num=250)
# x1 = [(-res.params['const']/res.params['geodesic_ratio'] - (res.params['euclidean_ratio']/res.params['geodesic_ratio'])*x) for x in x0]
# ax.plot(x0, x1, color=black)

# predict p(R) for a range of values
predicted_pr = np.zeros((len(x0), len(x0)))
for i, x in enumerate(x0):
    for ii, x2 in enumerate(x0):
        predicted_pr[i, ii] = glm_binom.predict(res.params, [x, x2, 1])

ax.imshow(predicted_pr, vmin=0, vmax=1, cmap='bwr', 
                        extent=[0, 1, 0, 1], origin='lower',
                    )


# style axes
ax.set(title="Logistic model predicting p(R) based on euclidean and geodesic ratio",
        xlabel='Geodesic ratio', ylabel='Euclidean ratio',
        xlim=[0,  1], 
        ylim=[0, 1])


clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'GLM_pR_prediction'), svg=True)



# %%
