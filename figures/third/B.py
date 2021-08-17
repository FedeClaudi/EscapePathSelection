# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

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
# %%
ROLLING_MEAN_WINDOW = 6

excluded = ['InfluenceZones']
cache_path = Path('/Users/federicoclaudi/Documents/Github/EscapePathSelection/cache/')

p_right = dict()
f, axes = plt.subplots(figsize=(12, 8), ncols=2, nrows=2, sharex=True, sharey=False)
for n, maze in enumerate(MAZES):
    for model, color in zip(MODELS, MODELS_COLORS):
        if model in excluded:
            continue

        # load data
        try:
            data_path = cache_path / f'{model}_training_on_{maze}.h5'
            data = pd.read_hdf(data_path, key='hdf')
        except Exception as e:
            print(f'Could not load {model} on {maze}', data_path, e, sep='\n')
            continue
        
        # get when mean success rate > criterion
        try:
            mean_sr = rolling_mean(data.play_status, ROLLING_MEAN_WINDOW)
        except AttributeError:
            logger.warning(f'{model} does not have play_status')
            continue

        mean_steps = rolling_mean(np.cumsum(data.play_steps), ROLLING_MEAN_WINDOW)
        try:
            above = np.where(mean_sr > .8)[0][0]
            above_steps = mean_steps[above]
        except IndexError:
            logger.info(f'{model} does not reach criterion')
            above = None
            above_steps = None

        # plot success rate
        plot_mean_and_error(mean_sr, 
                        rolling_mean(data.play_status_sem, ROLLING_MEAN_WINDOW), axes[0, n], label=model, color=color)

        # plot number of steps
        plot_mean_and_error(mean_steps, 
                        rolling_mean(data.play_steps_sem, ROLLING_MEAN_WINDOW), axes[1, n], label=model, color=color, err_alpha=.1)

        # plot p(R)
        plot_mean_and_error(rolling_mean(data.play_arm, ROLLING_MEAN_WINDOW), 
                        rolling_mean(data.play_arm_sem, ROLLING_MEAN_WINDOW), axes[0, n+1], label=model, color=color, err_alpha=.1)
        p_right[model] = (rolling_mean(data.play_arm, ROLLING_MEAN_WINDOW)[-1], rolling_mean(data.play_arm_sem, ROLLING_MEAN_WINDOW)[-1])

        # mark when mean-sr > criterion
        if above is not None:
            logger.info(f'|Maze {maze}| agent: {model} - above criterion at episode {above} ({above_steps:.2f} steps)')
            axes[0, n].plot([above, above], [0, mean_sr[above]], lw=3, color=color, ls=':')
            axes[1, n].plot([above, above], [0, mean_steps[above]], lw=3, color=color, ls=':')

        
    logger.debug('-'*20)
    axes[0, n].legend()
    break

axes[0, 0].set(title='M1', ylabel='accuracy', ylim=[0, 1])
axes[0, 1].set(ylim=[-0.1, 1.1], ylabel='p(R)')
axes[1, 0].set(xlabel='episodes', ylabel='comulative steps')
# axes[1, 1].set(xlabel='episodes')

clean_axes(f)
f.savefig(fig_3_path / 'panel_B_learning_curves.eps', format='eps', dpi=dpi)
plt.show()


# %%
for k, (mn, sm) in p_right.items():
    print(f'model: {k} final p(R): {mn:.2f} =- {sm:.2f}')
    



# %%
from rl import environment
a = pd.read_hdf('/Users/federicoclaudi/Documents/Github/EscapePathSelection/cache/DynaQ_20_training_on_M1.h5', key='hdf')

plt.plot(a.play_arm)
plt.plot(a.play_arm + a.play_arm_sem)
plt.plot(a.play_arm - a.play_arm_sem)
# %%
