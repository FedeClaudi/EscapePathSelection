# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from scipy.stats import fisher_exact
from rich import print
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from random import choice
from scipy.stats import sem

from fcutils.progress import track
from fcutils.plot.elements import plot_mean_and_error
from myterial import orange, salmon, grey_dark

from paper import Session, Stimuli

from figures._data_utils import get_recording_body_tracking, get_recording_maze_component
from figures._plot_utils import generate_figure, time_xax_params
from figures.colors import tracking_color, start_color, end_color
from figures.settings import fps, max_escape_frames, dpi
from figures.first import fig_1_path, M1, M2, M3, M4, M6

# %%

# get sessions
sessions = pd.DataFrame(Session().get_by(maze=4))

# get stimuli
stimuli = Stimuli.get_by_sessions(sessions)
stimuli = stimuli.loc[(stimuli.uid > 88)&(stimuli.uid < 100)]

# %%
axes = generate_figure(ncols=2, figsize=(16, 9))

ToT = {'left': [], 'right': [], 'center':[]}
center, left, right = 0, 0, 0
mice  = []
for i, stim in track(stimuli.iterrows(), description='Organizing data', total=len(stimuli)):
    tracking = get_recording_body_tracking(stim.recording_uid)
    if tracking is not None:
        start = stim.frame

        # get data after the stim
        tracking = tracking[start:start + max_escape_frames, :]
        if np.all(np.isnan(tracking)):
            continue
        

        # Get when the mouse leaves the threat platform
        try:
            allout = np.where(tracking[:, 1] > 85)[0]
            if len(allout) == 0:
                raise IndexError
        except IndexError:
            continue

        try:
            out = np.where(np.diff(allout) > 1)[0][-1]
        except IndexError:
            out = allout[0]

        if tracking[out, -1] <= 35:
            # speed below mean - 2*std: ignoring it
            continue
        if out > 4 * 40:
            # too long after stim onset
            continue

        x_out = tracking[out, 0]
        if x_out < 105:
            color = 'red'
            left += 1
            ToT['left'].append(out/40)
        elif x_out > 115:
            right += 1
            color='blue'
            ToT['right'].append(out/40)
        else:
            color = 'green'
            center += 1
            ToT['center'].append(out/40)
        mice.append(stim.mouse_id)

        axes[0].plot(tracking[:, 0], tracking[:, 1], color=color)
        axes[0].scatter(tracking[out, 0], tracking[out, 1], color='k', s=50, zorder=10)
        axes[1].scatter(out/40, tracking[out, -1], color=color, s=150)

tot = left + center + right

print(f'N trials {tot} | n mice {len(set(mice))}')
for arm, n in zip(('left', 'center', 'right'), (left, center, right)):
    print(f'p({arm}): {n/tot:.2f}')


# %%
# plot speed at ToT for all trials
speeds , tots= [], []
ax = generate_figure()
for dataset in (M1, M2, M3, M4, M6):
    for i, trial in dataset.iterrows():
        speeds.append(trial.speed[trial.out_of_t_frame - trial.stim_frame])
        tots.append(trial.time_out_of_t)

ax.hist(tots, bins=20)
# %%
mu = np.mean(speeds)
sdev = np.std(speeds)

print(f'Mean speed: {mu:.2f}, +- {sdev:.2f} - thershold: {mu -sdev:.2f}')
# %%
print(f'Mean ToT: ', {arm: f'{np.mean(v):.2f} +/- {np.std(v):.2f}' for arm,v in ToT.items()})
# %%
from scipy.stats import chisquare

observed = [left, center, right]
expected = [0.33 * tot, 0.33 * tot, 0.33 * tot]
chisquare(observed, expected)


# %%
