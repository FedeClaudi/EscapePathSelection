# %%
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.plot.distributions import plot_kde  # plot_distribution
from fcutils.maths.distributions import get_distribution


from figures.first import M1, M2, M3, M4,M6
from figures.second import fig_2_path
from figures._plot_utils import generate_figure
from figures.settings import dpi

from figures.bayes import Bayes

datasets = (M1, M2, M3, M4, M6)

# %%
'''
    Plot p(R) vs time in session
'''
bayes = Bayes()
axes = generate_figure(nrows=5, figsize=(16, 9), sharex=True)

cutoff = 60 * 60 * 1  # ignore trials after 2h of experiment
bins = np.linspace(0, cutoff, 30)
bin_width = 300  # seconds
bin_hw = int(bin_width / 2)
for ax, data in zip(axes, datasets):
    # bin trials by time in session
    data.trials['fps'] = [30 if t.uid < 184 else 40 for i,t in data.trials.iterrows()]
    data.trials['time_in_session'] = data.trials['stim_frame_session'] / data.trials['fps']
    data.trials = data.trials.loc[
            (data.trials.time_in_session <= cutoff) &
            (data.trials.escape_arm.isin(['left', 'right']))
            ]


    # plot pR as posterior for each bin
    X, trials_count = [], []
    for bin in bins:
        # get start and end of bin
        bstart = bin - bin_hw if (bin - bin_hw) > 0 else 0
        bend = bin + bin_hw if (bin + bin_hw) < cutoff else cutoff

        trials = data.trials.loc[(data.trials.time_in_session > bstart)&(data.trials.time_in_session < bend)]
        X.append(bin)

        N = len(trials)
        trials_count.append(N)
        if N < 3:
            continue
        nR = len(trials.loc[trials.escape_arm == 'right'])

        a, b, _, _, _, _, _ = bayes.grouped_bayes_analytical(N, nR)
        if a > 0  and b > 0:
            beta = get_distribution('beta', a, b, n_samples=100)

            plot_kde(
                    ax=ax, 
                    data=beta, 
                    vertical=True, 
                    z=bin, 
                    normto=bin_width * 0.55, 
                    color=data.color)

    ax.axhline(data.pR, lw=3, color=data.color, zorder=-1, alpha=.4)
    ax.set(ylim=[-.1, 1.1], ylabel=data.name)
axes[-1].set(xlabel='time (s)')
ax.figure.savefig(fig_2_path / 'timed_pR.eps', format='eps', dpi=600)
ax.figure.savefig(fig_2_path / 'timed_pR.svg', format='svg', dpi=600)

# %%

# %%
