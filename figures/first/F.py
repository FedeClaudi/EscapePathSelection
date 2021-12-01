
# %%
import sys
from pathlib import Path
import os
from rich import print
import numpy as np
import matplotlib.pyplot as plt
from myterial import salmon
from statsmodels.formula.api import glm as glm_api
import statsmodels.api as sm
from sklearn.model_selection import RepeatedStratifiedKFold as  KFold
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from loguru import logger

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from myterial import indigo, pink_dark, red, purple,  green_dark, teal_dark, blue

from figures.first import M1, M2, M3, M4,M6,  fig_1_path
from figures.glm import GLM
from figures._plot_utils import generate_figure
from figures.settings import dpi

datasets = (M1, M2, M3, M4, M6)



# %%
'''
    Fits a GLM with binomial distribution.
    Given the predicted p(R) of each trial, the accuracy
    is compared to the p(R) of each maze.

    The model(s) are tested on k-fold stratified cross validatoin
    which ensures that the same number of trials from each maze
    is conserved in the test set
'''

def fit(formula: str, data):
    '''
        It fits a single GLM based on the formula
        to a dataset and returns r squared
    '''
    # labels = pd.get_dummies(data.angle_ratio).T.iloc[0].values
    labels = data.maze.values

    # stratified (balanced) k-fold
    r_scores, models = [], []
    for train_index, test_index in KFold(n_splits=5, n_repeats=4).split(data, labels):
        x_train = data.iloc[train_index]
        x_test = data.iloc[test_index]

        glm_mod = glm_api(formula, x_train, family=sm.families.Binomial()).fit()
        glm_mod._x_train = x_train
        glm_mod._x_test = x_test

        # split by maze and compute pR for each maze
        x_test['maze'] = data.maze[x_test.index]
        pR = x_test.groupby('maze').outcomes.mean().to_dict()

        # pR of each trail in test set
        x_test['pR'] = [pR[maze] for maze in x_test.maze.values]

        # r^2 score of true pR vs predicted
        # r_scores.append(r2_score(x_test.pR, glm_mod.predict(x_test)))
        r_scores.append(pearsonr(x_test.pR, glm_mod.predict(x_test))[0])
        models.append(glm_mod)

    return r_scores, models

# %%
N_shuffles = 25

def sampling_k_elements(group, k=80):
    if len(group) < k:
        return group
    return group.sample(k)

# get GLM data  
glm_data = GLM.from_datasets(datasets).X

# M1-M6 balanced by count
balanced_data = glm_data.loc[glm_data.maze.isin(['M1', 'M6'])].reset_index(drop=True)

print(balanced_data.groupby('maze').count())

# GLM model formulas with subsets of the parameters
formulas = dict(
    full='outcomes ~ angle_ratio + geodesic_ratio + origin + time_in_session',
    no_angle = 'outcomes ~ geodesic_ratio + origin + time_in_session',
    no_geodesic = 'outcomes ~ angle_ratio + origin + time_in_session',
    no_origin = 'outcomes ~ angle_ratio + geodesic_ratio + time_in_session',
    no_time = 'outcomes ~ angle_ratio + geodesic_ratio + origin',
    trial='outcomes ~ origin + time_in_session',
    maze='outcomes ~ angle_ratio + geodesic_ratio',
)
results = {k:[] for k in formulas.keys()}
results_models = {k:[] for k in formulas.keys()}

shuffled_results = {k:np.zeros((N_shuffles, 20)) for k in formulas.keys()}

for fname, formula in formulas.items():
    # fit unshuffle data
    logger.info(f'Fitting GLM with formula "{fname}"')
    results[fname], results_models[fname] = fit(formula, glm_data)

    # fit shuffled data
    for rep in range(N_shuffles):
        shuffled = glm_data.copy()
        shuffled['outcomes'] = shuffled.outcomes.sample(frac=1).values

        shuffled_results[fname][rep, :], _ = fit(formula, shuffled)
logger.info('Done')

from scipy.stats import ttest_ind as ttest
from statsmodels.stats.multitest import multipletests

pvals, matched = [], []
for key in results.keys():
    if key == 'full': continue
    matched.append(('full', key))
    pvals.append(ttest(results['full'], results[key]).pvalue)


significant, pval, _, _ = multipletests(pvals, method='bonferroni', alpha=0.05)
different = {k[1]:v for k, v in zip(matched, significant)}
print(*zip(matched, significant))

# %%

colors = (indigo, pink_dark, red, purple,  green_dark, teal_dark, blue)
f = plt.figure(figsize=(16, 12))
axes = f.subplot_mosaic(
    '''
        AAAAAABBCCDD
        AAAAAAEEGGHH
        FFFFFFIIMMMM
    '''
)

# pearson correlation for real data
for n, (name, rsquares) in enumerate(results.items()):
    x = np.random.normal(n, .01, size=20)
    axes['A'].scatter(x, rsquares, s=60, color=colors[n], label=name)

# pearson correlation for shuffled data
for n, (name, rsquares) in enumerate(shuffled_results.items()):
    for rep in range(N_shuffles):
        x = np.random.normal(n, .02, size=20)
        axes['A'].scatter(x,  rsquares[rep], color=[.3, .3, .3], zorder=-1, alpha=.1)

# plot significants
for n, (name, significant) in enumerate(different.items()):
    if significant:
        axes['A'].plot([0, n+1], [1.1 + .02*n, 1.1 + .02*n], lw=2, color='k')

axes['A'].legend()
axes['A'].axhline(1, color='k', ls='--')
axes['A'].axhline(0, color='k', ls='--')
_ = axes['A'].set(ylabel='pearson correlation', xlabel='model', 
            xticks=np.arange(len(results)), xticklabels=results.keys(), ylim=[-1.1,1.3])

# ploy KDE stuff
kde_axes= 'BCDEGHILM'
for n, (name, rsquares) in enumerate(shuffled_results.items()):
    high = np.percentile(rsquares.ravel(), 95)
    sns.kdeplot(rsquares.flatten(), ax=axes[kde_axes[n]], shade=True, color=[.2, .2, .2])

    axes[kde_axes[n]].axvline(high, color='k', lw=2)

    for val in results[name]:
        axes[kde_axes[n]].scatter(val, 1, s=60, color=colors[n])
    axes[kde_axes[n]].set(title=name, ylim=[0, 3], xlim=[-1, 1], yticks=[0, 3])


# plot model parameters p values
params = {
    'angle_ratio':[],   
    'geodesic_ratio':[],    
    'origin':[],    
    'time_in_session':[], 
}
for model in results_models['full']:
    for name, val in model.pvalues.to_dict().items():
        if 'Intercept' == name:
            continue
        params[name].append(val)

for name, val in params.items():
    axes['F'].hist(np.log10(val), label=name, alpha=.5)


# plot model params magnitude
for param in params.keys():
    axes['M'].hist([mod.params[param] for mod in results_models['full']], label=param)

axes['F'].axvline(np.log10(0.05), lw=8, color='k')

axes['F'].set(xlabel='log(pvalue)')
_ = axes['F'].legend()
_ = axes['M'].legend()
axes['M'].set(title='parameters weights', ylabel='count')


f.tight_layout()
f.savefig(fig_1_path / 'glm_v2.eps', format='eps', dpi=dpi)


# %%
# print params magnitude
for param in params.keys():
    values = np.array([mod.params[param] for mod in results_models['full']])

    print(f'{param}: {np.mean(values):.2f} +/- {np.std(values):.2f}')