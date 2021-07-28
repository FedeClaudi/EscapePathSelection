# %%
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.plot.elements import ball_and_errorbar
from fcutils.maths.array import percentile_range
from fcutils.plot.distributions import plot_kde  # plot_distribution
from fcutils.maths.distributions import get_distribution

from myterial import blue_grey

from figures.first import M1, M2, M3, M4,M6,  fig_1_path
from figures.glm import GLM
from figures._plot_utils import generate_figure
from figures.settings import dpi
from figures._data_utils import get_pR_from_trials_df
from figures.colors import tracking_color
from figures.bayes import Bayes

datasets = (M1, M2, M3, M4, M6)
# %%
'''
    plot p(R) for naive vs normal trials
'''

# ----------------------------- p(R) naive vs nor ---------------------------- #
axes = generate_figure(ncols=5, figsize=(16, 8), sharex=True, sharey=True)

for ax, dataset in zip(axes, datasets):
    # get naive and not naive trials
    naive = dataset.trials.loc[dataset.trials.naive == 1]
    naive_mice = naive.mouse_id.unique()
    naive_trials_ids = []

    # only the first trial from each naive mouse
    for mouse in naive_mice:
        mouse_trials = naive.loc[naive.mouse_id == mouse] 
        naive_trials_ids.append(mouse_trials.index[0])

    naive_trials = dataset.trials.loc[naive_trials_ids]
    N = len(naive_trials)
    not_naive_trials = dataset.trials[~dataset.index.isin(naive_trials_ids)]
    

    print(f'Dataset: {dataset.name} | {dataset.n_trials} total trials of which {len(not_naive_trials)} are not naive and {N} are naive')

    # get a distribution of p(R) for randomly selected non naive trials
    if not N: continue
    random_samples = []
    for i in np.arange(3000):
        selected = not_naive_trials.sample(N)
        random_samples.append(get_pR_from_trials_df(selected))

    ax.hist(random_samples, color=blue_grey, label='not-naive trials', density=True, bins=10)
    prange = percentile_range(random_samples)
    ball_and_errorbar(
        prange.mean, 
        -.5, 
        ax,
        prange=prange,
        color = blue_grey,
        s=200,
        lw=4,
    )

    p = get_pR_from_trials_df(naive_trials)
    ax.plot(
        [p, p],
        [0, 1],
        lw=10,
        color = dataset.color,
        label = 'naive trials',
    )

    # ax.legend()
    ax.set(xlabel='p(R)', title=dataset.name)
    # break



axes[0].figure.suptitle('naive vs not naive p(R)')
axes[0].set(ylabel='density')
axes[1].axis('off')

ax.figure.savefig(fig_1_path / 'panel_G_histograms_more_bins.eps', format='eps', dpi=dpi)

