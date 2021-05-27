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
axes = generate_figure(ncols=5, figsize=(16, 8), sharex=False, sharey=True)

for ax, dataset in zip(axes, datasets):
    # get naive and not naive trials
    naive = dataset.trials.loc[dataset.trials.naive == 1]
    naive_mice = naive.mouse_id.unique()
    naive_trials_ids = []

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

    ax.hist(random_samples, color=blue_grey, label='not-naive trials', density=True, bins=5)
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
        [0, 6],
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

ax.figure.savefig(fig_1_path / 'panel_G_histograms.eps', format='eps', dpi=dpi)

# %%
'''
    Plot p(R) vs time in session
'''
bayes = Bayes()
axes = generate_figure(nrows=5, figsize=(16, 9), sharex=True)

cutoff = 60 * 60 * 1  # ignore trials after 2h of experiment
bins = np.linspace(0, cutoff, 10)
for ax, data in zip(axes, datasets):
    # bin trials by time in session
    data.trials['fps'] = [30 if t.uid < 184 else 40 for i,t in data.trials.iterrows()]
    data.trials['time_in_session'] = data.trials['stim_frame_session'] / data.trials['fps']
    data.trials = data.trials.loc[
            (data.trials.time_in_session <= cutoff) &
            (data.trials.escape_arm.isin(['left', 'right']))
            ]
    data.trials['bin'] = pd.cut(data.trials['time_in_session'], bins=bins)
    data.trials['sortkey'] = data.trials.bin.map(lambda x: x.left)
    data.trials = data.trials.sort_values('sortkey')

    # plot pR as posterior for each bin
    X, trials_count = [], []
    for bin in data.trials.bin.unique():
        trials = data.trials.loc[data.trials.bin == bin]
        window_size = bin.right - bin.left
        X.append(bin.left + window_size * .5)

        N = len(trials)
        trials_count.append(N)
        if N < 3:
            continue
        nR = len(trials.loc[trials.escape_arm == 'right'])

        a, b, _, _, _, _, _ = bayes.grouped_bayes_analytical(N, nR)
        if a > 0  and b > 0:
            beta = get_distribution('beta', a, b, n_samples=25000)

            plot_kde(
                    ax=ax, 
                    data=beta, 
                    vertical=True, 
                    z=bin.left + window_size*.5, 
                    normto=window_size * .75, 
                    color=data.color)

    # trials_count = np.array(trials_count)
    # ax.plot(X, trials_count / trials_count.max() / 3)    
    ax.axhline(data.pR, lw=3, color=data.color, zorder=-1, alpha=.4)
    ax.set(ylim=[-.1, 1.1], ylabel=data.name)
axes[-1].set(xlabel='time (s)')
ax.figure.savefig(fig_1_path / 'panel_G_time_binned_bayes.eps', format='eps', dpi=dpi)
# %%
