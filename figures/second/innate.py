# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.plot.elements import ball_and_errorbar
from fcutils.maths.array import percentile_range

from myterial import blue_grey

from figures.first import M1, M2, M3, M4, M6
from figures._plot_utils import generate_figure
from figures._data_utils import get_pR_from_trials_df
from figures.colors import tracking_color

# %%

# ----------------------------- p(R) naive vs nor ---------------------------- #
axes = generate_figure(ncols=5, figsize=(16, 8), sharex=False, sharey=True)
axes2 = generate_figure(ncols=3, nrows=2, figsize=(16, 9), sharex=False, sharey=True)

for ax, ax2, dataset in zip(axes, axes2, (M1, M2, M3, M4, M6)):
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
    for i in np.arange(10):
        selected = not_naive_trials.sample(N)
        random_samples.append(get_pR_from_trials_df(selected))

    ax.hist(random_samples, color=blue_grey, label='not-naive trials', density=True, bins=6)
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


    # plot tracking data
    for i, trial in dataset.iterrows():
        if i in naive_trials_ids:
            color, zorder = dataset.color, 2
        else:
            color, zorder = tracking_color, 1

        ax2.plot(trial.x, trial.y, color=color, zorder=zorder)

        ax2.axis('off')
        ax2.set(title=dataset.name)

axes[0].figure.suptitle('naive vs not naive p(R)')
axes[0].set(ylabel='density')
axes[1].axis('off')

axes2[-1].axis('off')
axes2[-1].axis('off')
# %%
