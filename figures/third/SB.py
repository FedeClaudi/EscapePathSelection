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
    Plot the  p(R) after training for eachmodel and maze
''' 



excluded = []

f, axes = plt.subplots(figsize=(12, 8), ncols=2, sharex=False, sharey=True)
for ax, maze in zip(axes, MAZES):
    logger.info('--'*20)
    xticks, xticklabels = [], []
    for n, (model, color) in enumerate(zip(MODELS, MODELS_COLORS)):
        if model in excluded:
            continue

        # load data
        try:
            data = pd.read_hdf(f'./cache/{model}_escape_on_{maze}.h5', key='hdf')
            if not len(data):
                raise ValueError
        except Exception as e:
            print(f'Could not load {model} on {maze}', f'./cache/{model}_escape_on_{maze}.h5', e, sep='\n')
            continue
        
        n_success = len(data.loc[data.escape_arm.isin(['left', 'right'])])

        logger.info(f'|maze {maze}| MODEL: "{model}"  p(success)={n_success/len(data) if n_success else 0}')

        if n_success == 0:
            continue

        n_l = len(data.loc[data.escape_arm == 'left']) / n_success
        n_r = len(data.loc[data.escape_arm == 'right']) / n_success

        if n_l + n_r > 1:
            raise ValueError

        x = list(np.array([-0.1, 0.1]) + n)
        ax.bar(x, [n_l, n_r], color=color, label=model, width=0.25)

        xticks.extend(x)
        xticklabels.extend(['l', 'r',])

    ax.set(xticks=xticks, xticklabels=xticklabels)
    ax.legend()

axes[0].set(title='M1', xlabel='episodes', ylabel='distance covered (a.u.)')
axes[1].set(title='M6', xlabel='episodes')
clean_axes(f)
f.suptitle('p(R)')

# f.savefig(fig_3_path / 'panel_SA_learning_curves_dist_trvld.eps', format='eps', dpi=dpi)
plt.show()


    

