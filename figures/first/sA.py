
# %%
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from figures._plot_utils import generate_figure
from figures.colors import tracking_color, tracking_color_dark
from figures.first import M4, fig_1_path
from figures.settings import trace_downsample, dpi

from paper import Tracking

'''
    Plot tracking during exploration for each session in M4
'''

# %%
from myterial import pink
import matplotlib.pyplot as plt

ax = generate_figure()

X, Y = [], []
skip = 40 * 120

for sess in list(M4.sessions):
    # Get tracking up to first stimulus
    first = M4.trials.loc[M4.trials.session_name == sess].iloc[0]

    tracking = pd.Series((Tracking.BodyPart & 'bpname="body"' & f'recording_uid="{first.recording_uid}"').fetch1())

    bad = np.where(tracking.speed > 150)

    x = tracking.x.copy()
    x[bad] = np.nan


    y = tracking.y.copy()
    y[bad] = np.nan

    X.extend(list(x[skip:first.stim_frame]))
    Y.extend(list(y[skip:first.stim_frame]))
    
hex = ax.hexbin(X, Y, mincnt=10, gridsize=50, bins='log', cmap='Blues')
cax = plt.axes([0.85, 0.1, 0.075, 0.8])

plt.colorbar(hex, cax=cax, label='Occupancy (s)', spacing='proportional')
ax.plot(
    x[skip:first.stim_frame][::trace_downsample * 2],
    y[skip:first.stim_frame][::trace_downsample * 2],
    color=pink, alpha=1, lw=3)

ax.axis('off')
ax.figure.savefig(fig_1_path / 'panel_S_A.eps', format='eps', dpi=dpi)
# %%
