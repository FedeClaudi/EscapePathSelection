
# %%
import sys
from pathlib import Path
import pandas as pd
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from figures._plot_utils import generate_figure
from figures.colors import tracking_color, tracking_color_dark
from figures._plot_utils import plot_threat_tracking_and_angle
from figures.first import M4, fig_1_path
from figures.settings import dpi

from paper import Tracking


'''
    Plot an example tracking of a full escape on M4 with body orientation

    Plot tracking on T in M4 and average orientaion on T for L vs R
'''

# %%
# plot tracking example
every = 8
trial = M4.trials.iloc[102]


body = pd.DataFrame(Tracking.BodyPart & f'recording_uid="{trial.recording_uid}"' & 'bpname="body"').iloc[0]
snout = pd.DataFrame(Tracking.BodyPart & f'recording_uid="{trial.recording_uid}"' & 'bpname="snout"').iloc[0]
tail = pd.DataFrame(Tracking.BodyPart & f'recording_uid="{trial.recording_uid}"' & 'bpname="tail_base"').iloc[0]

start = trial.stim_frame
end = trial.at_shelter_frame

ax = generate_figure()

ax.plot(trial.x, trial.y, lw=8, color=tracking_color, alpha=.4)

for bp1, bp2 in  ((body, snout), (body, tail)):
    ax.plot(
        [bp1.x[start:end:every], bp2.x[start:end:every]],
        [bp1.y[start:end:every], bp2.y[start:end:every]],
        color = tracking_color_dark, lw=4, solid_capstyle='round')

ax.scatter(snout.x[start:end:every], snout.y[start:end:every], s=200, zorder=200, color=tracking_color_dark)

ax.axis('off')
ax.figure.savefig(fig_1_path / 'panel_B_tracking_example.eps', format='eps', dpi=dpi)

# %%
# plot average heading while on threat platform

axes = plot_threat_tracking_and_angle(M4)
axes[0].figure.savefig(fig_1_path / 'panel_B_angles.eps', format='eps', dpi=dpi)

# %%
