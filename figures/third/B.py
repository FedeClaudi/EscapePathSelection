# %%
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

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

import warnings
warnings.filterwarnings("ignore")

'''
    Plot the taining curves of all models (p(success))
''' 
# %%
ROLLING_MEAN_WINDOW = 6

excluded = ['InfluenceZones']
cache_path = Path('/Users/federicoclaudi/Documents/Github/EscapePathSelection/cache/')

p_right = dict()
f, axes = plt.subplots(figsize=(16, 14), ncols=3, nrows=4, sharex=False, sharey=False)

for n, maze in enumerate(MAZES):
    for model_n, (model, color) in enumerate(zip(MODELS, MODELS_COLORS)):
        if model in excluded:
            continue

        # load data
        try:
            data_path = cache_path / f'{model}_training_on_{maze}.h5'
            data = pd.read_hdf(data_path, key='hdf')
        except Exception as e:
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
            above = None
            above_steps = None

        # plot success rate
        plot_mean_and_error(mean_sr, 
                        rolling_mean(data.play_status_sem, ROLLING_MEAN_WINDOW), axes[0, n], label=model, color=color)

        # plot number of steps to above threshold
        if above_steps is not None:
            axes[1, n].bar(model_n, above_steps, label=model, color=color)

        # plot p(R)
        pR = rolling_mean(data.play_arm, ROLLING_MEAN_WINDOW)
        plot_mean_and_error(pR, 
                        rolling_mean(data.play_arm_sem, ROLLING_MEAN_WINDOW), axes[2, n], label=model, color=color, err_alpha=.1)
        p_right[model] = (pR[-1], rolling_mean(data.play_arm_sem, ROLLING_MEAN_WINDOW)[-1])

        # plot p(R) * accuracy
        axes[3, n].plot(pR*mean_sr, color=color)

        # mark when mean-sr > criterion
        if above is not None:
            axes[0, n].plot([above, above], [0, mean_sr[above]], lw=3, color=color, ls=':')

        
    axes[0, n].legend()


axes[0, 0].set(title='M1', ylabel='accuracy', ylim=[0, 1])
axes[2, 0].set(ylim=[-0.1, 1.1], ylabel='p(R)', xticks=[])
axes[1, 0].set(xlabel='Model', ylabel=r'steps to 80% accuracy', xticks=[0, 1, 2], xticklabels=MODELS, yticks=[0, 100000])
axes[3, 0].set(xlabel='episodes', ylabel='p(R)*accuracy')
axes[0, 1].set(title='M2', ylim=[0, 1])
axes[0, 2].set(title='M3', ylim=[0, 1])

clean_axes(f)
f.savefig(fig_3_path / 'panel_B_learning_curves.eps', format='eps', dpi=dpi)
plt.show()



# %%
# plot barplot of steps to 80%

maze = 'M1'

f, axes = plt.subplots(figsize=(16, 9), ncols=2)

for n, (model, color) in enumerate(zip(MODELS, MODELS_COLORS)):
    data_path = cache_path / f'{model}_training_on_{maze}.h5'
    data = pd.read_hdf(data_path, key='hdf')
    mean_sr = rolling_mean(data.play_status, ROLLING_MEAN_WINDOW)

    try:
        above = np.where(mean_sr > .8)[0][0]
        above_steps = mean_steps[above]
    except IndexError:
        logger.info(f'{model} does not reach criterion')
    else:
        axes[0].bar(n, above_steps)



# %%
for k, (mn, sm) in p_right.items():
    print(f'model: {k} final p(R): {mn:.2f} =- {sm:.2f}')
    


# %%
a = pd.read_hdf('/Users/federicoclaudi/Documents/Github/EscapePathSelection/cache/guided_exploration.h5', key='hdf')

print(
    np.nanmean(a.iloc[1].escape_arms),
    np.nanmean(a.iloc[2].escape_arms),
)
# %%
