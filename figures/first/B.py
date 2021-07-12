
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

# keep only trials from dead end experiments
M4.trials = M4.trials.loc[(M4.trials.uid > 88)&(M4.trials.uid < 100)]
print(len(M4.trials))

axes, L, R = plot_threat_tracking_and_angle(M4, catwalk_only=False)
axes[0].figure.savefig(fig_1_path / 'panel_B_angles.eps', format='eps', dpi=dpi)

# %%
import matplotlib.pyplot as plt
import numpy as np

f = plt.figure(figsize=(16, 9))
ax1 = f.add_subplot(121, projection='polar')
ax2 = f.add_subplot(122, projection='polar')
ax1.set_theta_zero_location("N")
ax2.set_theta_zero_location("N")
ax1.set_theta_direction(-1)
ax2.set_theta_direction(-1)

nframes = len(L.orientation.iloc[0])
for step in range(nframes):
    l = np.mean([orientation[step-5:step] for orientation in L.orientation])
    r = np.mean([orientation[step-5:step] for orientation in R.orientation])

    if step % 5 == 0:
        width = 0.1 
        alpha = 1 * (step / nframes)
    else:
        continue

    ax1.arrow(l/180.*np.pi, 0.5, 0, 1, alpha = alpha, width = width,
                 edgecolor = 'black', facecolor = tracking_color,zorder = 5)
    ax2.arrow(r/180.*np.pi, 0.5, 0, 1, alpha = alpha, width = width,
                 edgecolor = 'black', facecolor = tracking_color, zorder = 5)

ax1.set(title='left trials', yticks=[])
ax2.set(title='right trials', yticks=[])
ax.figure.savefig(fig_1_path / 'panel_B_angles_POLAR.eps', format='eps', dpi=dpi)

# %%

