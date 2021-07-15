import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from fcutils.plot.figure import clean_axes
from fcutils.plot.elements import plot_mean_and_error
from fcutils.maths import rolling_mean

import sys
sys.path.append('./')

from figures.third import MODELS_COLORS, MODELS, MAZES, fig_3_path
from figures.settings import dpi
'''
    Plot the taining curves of all models (p(success))
''' 

ROLLING_MEAN_WINDOW = 6

# excluded = ['DynaQ_30', 'InfluenceZones']

excluded = ['InfluenceZones']

f, axes = plt.subplots(figsize=(12, 8), ncols=2, nrows=2, sharex=True, sharey=False)
for n, maze in enumerate(MAZES):
    for model, color in zip(MODELS, MODELS_COLORS):
        if model in excluded:
            continue

        # load data
        try:
            data = pd.read_hdf(f'./cache/{model}_training_on_{maze}.h5', key='hdf')
        except Exception as e:
            print(f'Could not load {model} on {maze}', f'./cache/{model}_training_on_{maze}.h5', e, sep='\n')
            continue
        
        # get when mean success rate > criterion
        mean_sr = rolling_mean(data.success, ROLLING_MEAN_WINDOW)
        mean_steps = rolling_mean(np.cumsum(data.n_steps), ROLLING_MEAN_WINDOW)
        try:
            above = np.where(mean_sr > .85)[0][0]
            above_steps = mean_steps[above]
        except IndexError:
            above = None

            above_steps = None

        # plot success rate
        plot_mean_and_error(mean_sr, 
                        rolling_mean(data.success_sem, ROLLING_MEAN_WINDOW), axes[0, n], label=model, color=color)

        # plot number of steps
        plot_mean_and_error(mean_steps, 
                        rolling_mean(data.n_steps_sem, ROLLING_MEAN_WINDOW), axes[1, n], label=model, color=color, err_alpha=.1)

        # mark when mean-sr > criterion
        if above is not None:
            logger.info(f'|Maze {maze}| agent: {model} - above criterion at episode {above} ({above_steps:.2f} steps)')
            axes[0, n].plot([above, above], [0, mean_sr[above]], lw=3, color=color, ls=':')
            axes[1, n].plot([above, above], [0, mean_steps[above]], lw=3, color=color, ls=':')

    logger.debug('-'*20)
    axes[0, n].legend()

axes[0, 0].set(title='M1', ylabel='accuracy', ylim=[0, 1])
axes[0, 1].set(title='M6', ylim=[0, 1])
axes[1, 0].set(xlabel='episodes', ylabel='comulative steps', ylim=[0, 15000])
axes[1, 1].set(xlabel='episodes', ylim=[0, 15000])

clean_axes(f)
f.suptitle('C | success training curve')
# f.savefig(fig_3_path / 'panel_C_learning_curves.eps', format='eps', dpi=dpi)
plt.show()




    

