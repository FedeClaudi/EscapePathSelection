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

        
f, axes = plt.subplots(ncols=2, figsize=(16, 9), sharex=False, sharey=True)

for n, maze in enumerate(data.maze.unique()):
    for mn, model in enumerate(data.model.unique()):
        print(model)
        exp = np.array(data.loc[(data.model == model)&(data.maze == maze)].results.iloc[0])

        mean = np.mean(np.cumsum(exp, 0), 1)
        std = np.std(np.cumsum(exp, 0), 1)
        plot_mean_and_error(mean, std, axes[n], lw=4, color=MODELS_COLORS[mn], label=model)

axes[0].legend()
axes[1].legend()
axes[0].set(title='M1', ylabel='comulative successs', xlabel='number of sessions')
axes[1].set(title='M6', xlabel='number of sessions')

clean_axes(f)
f.savefig(fig_3_path / 'panel_E_pR.eps', format='eps', dpi=dpi)
plt.show()




    



# %%
