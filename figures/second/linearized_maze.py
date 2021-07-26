# %%
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from myterial.utils import make_palette

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.plot.figure import clean_axes
from fcutils.maths.coordinates import cart2pol
from myterial import salmon

from paper import Explorations

from figures.first import M1
from figures.settings import dpi
from figures.second import fig_2_path


# %%
sess_number = 27  # 34 49 23
exploration = pd.Series((Explorations & f'session_name="{M1.sessions[sess_number]}"').fetch1())

trial = M1.trials.loc[M1.trials.session_name == M1.sessions[sess_number]].iloc[0]


shelter_color = '#F695BD'
threat_color = '#3B5D70'
arm_color = '#F0F1E2'
arm_color_dark = '#D8D9C7'

right_palette = make_palette(threat_color, arm_color, 35) + make_palette(arm_color, arm_color, 75) + make_palette(arm_color, shelter_color, 70)
left_palette = make_palette(shelter_color, arm_color_dark, 70) + make_palette(arm_color_dark, arm_color_dark, 75) + make_palette(arm_color_dark, threat_color, 35)

custom_palette = right_palette + left_palette


# get position centered at the origin
center = (500, 500)

def get_theta_and_time(x, y):
    theta = cart2pol(x, y)[1] + 180 + 274
    theta[theta > 360] -= 360
    time = np.arange(len(x)) / exploration.fps / 60
    return theta, time

# get exploration data
x = exploration.body_tracking[:, 0] - center[0]
y = exploration.body_tracking[:, 1] - center[1]
theta, time = get_theta_and_time(x, y)
colors = [custom_palette[int(t)] for t in theta]

# get trial data
trial_x, trial_y = trial.x / 0.22 - center[0], trial.y / 0.22 - center[1]
trial_theta, trial_time = get_theta_and_time(trial_x, trial_y)
trial_time += time[-1]

# plot exploration
f, axes = plt.subplots(ncols=2, figsize=(20, 8), gridspec_kw={'width_ratios': [2, 3]})
# axes[0].plot(x, y, lw=2, color=[.4, .4, .4])

# axes[0].scatter(x, y, color=[.2, .2, .2], s=45, zorder=99,)
# axes[0].scatter(x, y, c=colors, s=40, zorder=100,)
axes[0].plot(x, y, color=[.3, .3, .3],lw=2, zorder=100,)


# axes[1].scatter(time, theta, color=[.2, .2, .2], s=65, zorder=99)
# axes[1].scatter(time, theta, c=colors, s=60, zorder=100)
axes[1].plot(time, theta, color=[.2, .2, .2], lw=2, zorder=100)

axes[1].axhline(180, ls=':', zorder=-1, color=[.3, .3, .3])

# plot trial
axes[0].plot(trial_x, trial_y, color=salmon, lw=8, zorder=200)
axes[1].plot(trial_time, trial_theta, color=salmon, lw=8, zorder=200)

# style
axes[0].axis('off')
axes[1].set(xlabel='time (min)', yticks=[0, 90, 180, 270, 360], 
            yticklabels=['threat', 'right arm', 'shelter', 'left arm', 'threat'])

clean_axes(f)
axes[0].figure.savefig(fig_2_path / 'linearized_maze_exploration.eps', format='eps', dpi=600)


# tstart, tstop = 200, 500
# axes[0].plot(x[tstart:tstop], y[tstart:tstop], color=[.2, .2, .2], lw=6)
# axes[1].plot(time[tstart:tstop], theta[tstart:tstop], color=[.2, .2, .2], lw=6)


# %%


# %%
