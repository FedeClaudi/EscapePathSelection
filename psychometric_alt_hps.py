"""
Analysis script for psychometric maze excluding alernative hypotheses:
        - Exploration
        - outward path
        - orientation
        - position
"""


# %%
# Imports
import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point, ball_and_errorbar
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.dbase.TablesDefinitionsV4 import *


# %%
# --------------------------------- Load data -------------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = 1, 
    tracking = 'all'
)

trials = TrialsLoader(**params)
trials.load_psychometric()


# ---------------------------------- cleanup --------------------------------- #
"""
    In some trials on M1 mice go left first and then right, discard these trials.
"""
goodids, skipped = [], 0
_trials = trials.datasets['maze1']
for i, trial in _trials.iterrows():
    if trial.escape_arm == "left":
        if np.max(trial.body_xy[:, 0]) > 600:
            skipped += 1
            continue
    goodids.append(trial.stimulus_uid)

t = trials.datasets['maze1'].loc[trials.datasets['maze1'].stimulus_uid.isin(goodids)]
trials.datasets['maze1'] = t



# %%
# ---------------------------------------------------------------------------- #
#                             EXPLORATION ANALYSIS                             #
# ---------------------------------------------------------------------------- #
# ------------------------- Get sessions for datasets ------------------------ #
datasets_sessions = trials.get_datasets_sessions()
dataset_explorations = {}

for ds, sessions in datasets_sessions.items():
    dataset_explorations[ds] = []
    for sess in sessions['session_name']:
        dataset_explorations[ds].append((Explorations & f"session_name='{sess}'").fetch1())

dataset_explorations = {ds: pd.DataFrame(d) for ds, d in dataset_explorations.items()}

# %%
# ----------------------- Look at exploration duration ----------------------- #
f, axarr = plt.subplots(ncols = 2)


for n, (ds, data) in enumerate(dataset_explorations.items()):
    ball_and_errorbar(n, data.duration_s.mean(), data.duration_s.values, axarr[0], orientation='vertical',
                    s=250, color=paper.maze_colors[ds])
    
    ball_and_errorbar(n, data.distance_covered.mean(), data.distance_covered.values, axarr[1], 
                    orientation='vertical',
                    s=250, color=paper.maze_colors[ds])

axarr[0].set(title="mean exploration duration", ylabel="duration (s)", xlabel="maze",
                xticks=[0, 1, 2, 3, 4], xticklabels=paper.five_mazes)
axarr[1].set(title="mean exploration distance covered", ylabel="distance (px)", xlabel="maze",
                xticks=[0, 1, 2, 3, 4], xticklabels=paper.five_mazes)
clean_axes(f)



# %%
# ----------------- Look at ammount of time spend on each arm ---------------- #




