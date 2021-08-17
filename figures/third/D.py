# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from fcutils.plot.figure import clean_axes
from fcutils.plot.elements import plot_mean_and_error
from fcutils.maths import rolling_mean

import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')



from figures.third import MODELS_COLORS, MODELS, MAZES, fig_3_path
from figures.settings import dpi

'''
    Plot comulative successes for models trained on guided epxloration
''' 

ROLLING_MEAN_WINDOW = 21


data = pd.read_hdf(f'/Users/federicoclaudi/Documents/Github/EscapePathSelection/cache/guided_exploration.h5', key='hdf')
print(data)

    



# %%
f, ax = plt.subplots(figsize=(12, 8))

for mn, (model, color) in enumerate(zip(data.model.unique(), MODELS_COLORS)):
    print(model)
    exp = np.array(data.loc[(data.model == model)&(data.maze == 'M1')].results.iloc[0])

    mean = np.mean(exp, 1)
    for n, mu in enumerate(mean):
        if mu>.8:
            y = .5+ mn * 0.025
        else:
            y =  mn * 0.025

        ax.scatter(n, y, color=color, s=150)
ax.set(yticks=[0.025, .525], yticklabels=['fail', 'success'], xlabel='session #')
clean_axes(f)
f.savefig(fig_3_path / 'panel_D_guided_expl_successes.eps', format='eps', dpi=dpi)
# %%


'''
    Get p(R)
'''
from scipy.stats import sem
for i, row in data.loc[data.maze=='M1'].iterrows():
    arms = [arm if np.mean(success > 0.8) else np.nan for success, arm in zip(row.results, row.escape_arms) ]

    mn = np.nanmean(arms)
    sm = sem(arms, nan_policy='omit')
    print(f'model: {row.model} p(R): {mn:.2f} +- {sm:2f}')

# %%
