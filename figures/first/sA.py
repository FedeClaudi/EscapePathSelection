
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


# %%
ax = generate_figure()

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

    ax.plot(
        x[skip:first.stim_frame][::trace_downsample * 2],
        y[skip:first.stim_frame][::trace_downsample * 2],
        color=tracking_color, alpha=.25, lw=1.25)

ax.plot(
    x[skip:first.stim_frame][::trace_downsample * 2],
    y[skip:first.stim_frame][::trace_downsample * 2],
    color=tracking_color_dark, alpha=1, lw=2.25)

ax.axis('off')
ax.figure.savefig(fig_1_path / 'panel_S_A.eps', format='eps', dpi=dpi)
# %%
