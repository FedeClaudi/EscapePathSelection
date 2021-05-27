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
from tqdm import tqdm
from random import sample
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point, plot_line_outlined
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
    lights = None,
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

euclidean_dists, _, _ = get_euclidean_dist_for_dataset(trials.datasets, trials.shelter_location)
mazes = {k:m for k,m in _mazes.items() if k in paper.five_mazes}

mazes_stats = pd.DataFrame(dict(
    maze = list(mazes.keys()),
    geodesic_ratio = [v['ratio'] for v in mazes.values()],
    euclidean_ratio = list(euclidean_dists.values()),
    angles_ratio = [m['left_path_angle']/m['right_path_angle'] for m in mazes.values()]
))
mazes_stats



# %%

# ---------------------------------------------------------------------------- #
#                                 GLM MODELLING                                #
# ---------------------------------------------------------------------------- #

# ------------------------------ Fit on all MICE ----------------------------- #

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
"""
    Fit GLM on each mouse?

    for each session get p(R) (from bayes) and ratios -> 
    predict p(R) with GLM
"""
data = dict(
    maze = [],
    geodesic_ratio = [],
    euclidean_ratio = [],
    pr = [],
)

for i, ds in hierarchical_pRs.iterrows():
    geo = mazes_stats.loc[mazes_stats.maze == ds.dataset].geodesic_ratio.values[0]
    euc = mazes_stats.loc[mazes_stats.maze == ds.dataset].euclidean_ratio.values[0]
    for pr in ds.means:
        data['maze'].append(ds.dataset)
        data['geodesic_ratio'].append(geo)
        data['euclidean_ratio'].append(euc)
        data['pr'].append(pr)

data = pd.DataFrame(data)
data.head()


# %%


# Fit model
X = data[['maze', 'geodesic_ratio', 'euclidean_ratio']]
X = sm.add_constant(X, prepend=False)
Y = data[['pr']]

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=.33)

if len(xtest.maze.unique()) < 5 or len(xtrain.maze.unique()) < 5:
    raise ValueError('Wish I had more data man')

xtrain = xtrain.drop(columns='maze')
testmaze = xtest.maze.values
xtest = xtest.drop(columns='maze')


glm_binom = sm.GLM(ytrain, xtrain[['geodesic_ratio', 'euclidean_ratio', 'const']], 
                    family=sm.families.Binomial())
res = glm_binom.fit()
res.summary()


f, ax = plt.subplots(figsize=(12, 12))

for maze, pr, prhat in zip(testmaze, ytest.pr, res.predict(xtest[['geodesic_ratio', 'euclidean_ratio', 'const']])):
    ax.scatter(pr, prhat, 
                        color=paper.maze_colors[maze], 
                        s=200, zorder=1, ec='k')


ax.plot([0, 1], [0, 1], zorder=-1, ls=':', color=[.6, .6, .6])

for i, row in grouped_pRs.iterrows():
    vline_to_point(ax, row['mean'], row['mean'], color=paper.maze_colors[row.dataset], zorder=-1)


ax.set(xlabel='Actual p(R)', ylabel='predicted p(R)')
clean_axes(f)

save_figure(f, os.path.join(paths.plots_dir, 'GLM_mice_crossval'))



# %%
# ---------------------------------------------------------------------------- #
#                               FIT ON ALL TRIALS                              #
# ---------------------------------------------------------------------------- #
"""
    FIT BERNOULLI GLM ON ALL TRIALS
        at each iteration draw a random sample of trials to train on, 
        get p(R) for ecah maze for the train and test sets, 
        train to predict p(R) from maze on train and evaluate on test.
"""
predictors = ['geodesic_ratio']

# ----------------------- Get a DF with all the trials ----------------------- #
data = {'outcome':[], 'maze':[]}
for maze, trs in trials.datasets.items():
    trs = trs.loc[trs.escape_arm != 'center']

    for i, trial in trs.iterrows():
        if trial.escape_arm == 'left':
            data['outcome'].append(0)
        else:
            data['outcome'].append(1)
        data['maze'].append(maze)
data = pd.DataFrame(data)
ntrials = len(data)


# ------------------------------ iterate N times ----------------------------- #
coeffs = {pred: [] for pred in predictors}
coeffs['const'] = []
TRUE, PREDICTION = [], []
R2 = []
f, ax = plt.subplots(figsize=(10, 10))
for i in tqdm(np.arange(500)):
    # Split train/test
    trainidx = sample(list(np.arange(ntrials)), int(ntrials * .66))
    testidx = [i for i in np.arange(ntrials) if i not in trainidx]

    train = data.iloc[trainidx]
    test = data.iloc[testidx]

    # ----------------------- Summarise data for train/test ---------------------- #
    train_data = {'maze':[], 'pr':[], 'geodesic_ratio':[], 'euclidean_ratio':[], 'angles_ratio':[]}
    test_data = {'maze':[], 'pr':[], 'geodesic_ratio':[], 'euclidean_ratio':[], 'angles_ratio':[]}

    for maze in trials.datasets.keys():
        tr = train.loc[train.maze == maze]
        te = test.loc[test.maze == maze]

        geo = mazes_stats.loc[mazes_stats.maze == maze].geodesic_ratio.values[0]
        euc = mazes_stats.loc[mazes_stats.maze == maze].euclidean_ratio.values[0]
        ang = mazes_stats.loc[mazes_stats.maze == maze].angles_ratio.values[0]

        train_data['maze'].append(maze)
        train_data['pr'].append(tr.outcome.sum() / len(tr))
        train_data['geodesic_ratio'].append(geo)
        train_data['euclidean_ratio'].append(euc)
        train_data['angles_ratio'].append(ang)


        test_data['maze'].append(maze)
        test_data['pr'].append(te.outcome.sum() / len(te))
        test_data['geodesic_ratio'].append(geo)
        test_data['euclidean_ratio'].append(euc)
        test_data['angles_ratio'].append(ang)
        
    test = pd.DataFrame(test_data)
    train = pd.DataFrame(train_data)

    # ------------------------------------ Fit ----------------------------------- #
    exog = train[predictors]
    exog = sm.add_constant(exog, prepend=False)
    endog = train[['pr']]

    testexog = test[predictors]
    testexog = sm.add_constant(testexog, prepend=False)

    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Plot predictions
    ax.scatter(train.pr, res.predict(exog), color=[.4, .4, .4], alpha=.8)
    ax.scatter(test.pr, res.predict(testexog), zorder=99,
            c=[paper.maze_colors[m] for m in trials.datasets.keys()], alpha=.6)

    TRUE.extend(list(test.pr))
    PREDICTION.extend(list(res.predict(testexog)))
    R2.append(r2_score(list(test.pr), list(res.predict(testexog))))

    for par, val in res.params.iteritems():
        coeffs[par].append(val)

ax.plot([0, 1], [0, 1], ls='--', color=[.8, .8, .8], zorder=-1)

for i, row in grouped_pRs.iterrows():
    vline_to_point(ax, row['mean'], row['mean'], color=paper.maze_colors[row.dataset], zorder=-1)
    ax.scatter(row['mean'], row['mean'], color='white',  alpha=1,
                        lw=1, s=150, ec='w', zorder=101)
    ax.scatter(row['mean'], row['mean'], color=desaturate_color(paper.maze_colors[row.dataset]), 
                        lw=1, s=100, ec=[.2, .2, .2], zorder=101)

    ax.text(row['mean'], -.03, row.dataset.replace('maze', 'M'), 
                    fontsize=14, horizontalalignment='center')


mse = round(mean_squared_error(TRUE, PREDICTION), 4)
ax.set(title=f'test set predictions [mse: {mse}]\n predictors: {predictors}', xlabel='actual p(R)', ylabel='predicted p(R)')
clean_axes(f)

save_figure(f, os.path.join(paths.plots_dir, 'GLM_trials_crossval'), svg=True)


# ---------------------------- Plot coeffs spread ---------------------------- #

f, ax = plt.subplots()
for coeff, vals in coeffs.items():
    ax.hist(vals, label=coeff)
ax.legend()

ax.set(title='GLM coefficients', xlabel='value', ylabel='count')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'GLM_trials_crossval_coefffs'))



# %%

# ------------------------------ Plot R2 scores ------------------------------ #
f, ax = plt.subplots(figsize=(10, 10))

ax.hist(R2, bins=20, color=[.6, .6, .6], density=True, zorder=-10)
plot_line_outlined(ax, [np.mean(R2), np.mean(R2)], [0, 4.5], outline_color='w', color=[.2, .2, .2], lw=4, outline=0)

ax.set(ylabel='density', xlabel='$r**2$')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'GLM_trials_r_squared_geoonly'), svg=True)
# %%
