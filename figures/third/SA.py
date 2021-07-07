import pandas as pd
import matplotlib.pyplot as plt

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

excluded = ['DynaQ_30', 'InfluenceZones']

f, axes = plt.subplots(figsize=(12, 8), ncols=2, sharex=True, sharey=True)
for ax, maze in zip(axes, MAZES):
    for model, color in zip(MODELS, MODELS_COLORS):
        if model in excluded:
            continue

        # load data
        try:
            data = pd.read_hdf(f'./cache/{model}_training_on_{maze}.h5', key='hdf')
        except Exception as e:
            print(f'Could not load {model} on {maze}', f'./cache/{model}_training_on_{maze}.h5', e, sep='\n')
            continue
        
        plot_mean_and_error(rolling_mean(data.distance_travelled, ROLLING_MEAN_WINDOW), 
                        rolling_mean(data.distance_travelled_sem, ROLLING_MEAN_WINDOW), ax, label=model, color=color)
    ax.legend()

axes[0].set(title='M1', xlabel='episodes', ylabel='distance covered (a.u.)')
axes[1].set(title='M6', xlabel='episodes')
clean_axes(f)
f.suptitle('C | distance travelled training curve')

f.savefig(fig_3_path / 'panel_SA_learning_curves_dist_trvld.eps', format='eps', dpi=dpi)
plt.show()


    

