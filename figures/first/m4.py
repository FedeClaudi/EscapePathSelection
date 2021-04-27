'''
    Shows how mice escape in response to stimuli
'''
# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from random import choice

from fcutils.progress import track
from myterial import orange

from paper import Session, Stimuli, TrackingData, Mouse

from figures._data_utils import get_recording_body_tracking, get_recording_maze_component
from figures._plot_utils import generate_figure
from figures.colors import tracking_color, start_color, end_color
from figures.settings import max_escape_duration, fps, max_escape_frames

# ------------------------------- get all data ------------------------------- #

# get sessions
sessions = Session().get_by(maze=4)

# get stimuli
stimuli = Stimuli.get_by_sessions(sessions)


# stack data around stimuli
stimuli_data, random_data = [], []
sessions, mice = [], []
stim_end_on_shelter = []
random_end_on_shelter = []
for i, stim in track(stimuli.iterrows(), description='Organizing data', total=len(stimuli)):
    tracking = get_recording_body_tracking(stim.recording_uid)
    if tracking is not None:
        start = stim.frame

        # get data after the stim
        stimuli_data.append(tracking[start:start + max_escape_frames, :])

        # get random time from when mouse on T
        maze_component = get_recording_maze_component(stim.recording_uid)
        on_T = np.where(maze_component==1)[0]
        on_T = on_T[np.where((120 * fps < on_T)&(on_T < len(tracking) - 120 * fps))]
        random_start = choice(on_T)
        random_data.append(tracking[random_start:random_start + max_escape_frames, :])

        sessions.append(stim.session_name)
        mice.append(stim.mouse_id)

        # check if mouse at threat at end of trials
        stim_end_on_shelter.append(1 if maze_component[start + max_escape_frames] == 0 else 0)
        random_end_on_shelter.append(1 if maze_component[random_start + max_escape_frames] == 0 else 0)


# print relevant numbers
sessions = set(sessions)
mice = set(mice)
logger.info(f'[{orange}]{__name__}: Got data from {len(mice)} mice in {len(sessions)} sessions. {len(stimuli_data)} trials in total')
logger.info(f'[green]p(shelter) | stim evoked: {np.mean(stim_end_on_shelter):.3f} | random {np.mean(random_end_on_shelter):.3f}')
# stack data
stimuli_data = np.vstack(stimuli_data).reshape(-1, max_escape_frames, 3)  # n trials x n frames x 3
random_data = np.vstack(random_data).reshape(-1, max_escape_frames, 3)

# %%
# ------------------------------- plot tracking ------------------------------ #
axes = generate_figure(ncols=2, figsize=(16, 8))
for ax, data in zip(axes, (stimuli_data, random_data)):
    ax.plot(data[:, :, 0].T, data[:, :, 1].T, lw=1, alpha=.8, color=tracking_color)
    ax.scatter(data[:, 0, 0], data[:, 0, 1], s=30, color=start_color, zorder=10, label='start')
    ax.scatter(data[:, -1, 0], data[:, -1, 1], s=30, color=end_color, zorder=10, label='end')

    ax.axis('off')

axes[0].set(title='Stimuli tracking')
_ = axes[1].set(title='Random tracking')
# %%

# ----------------------------- plot speed traces ---------------------------- #

ax = generate_figure()
ax.plot(np.nanmean(stimuli_data[:, :, 2], 0))
ax.plot(np.nanmean(random_data[:, :, 2], 0))
# TODO plot nicely


# %%

# ------------------------------ test p(shelter) ----------------------------- #
# TODO test and print stuff