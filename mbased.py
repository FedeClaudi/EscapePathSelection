
# %%
# Imports
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.colors import *
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.dbase.TablesDefinitionsV4 import Explorations, Session, Stimuli, TrackingData


# TODO deal with sessions not being registered properly
# TODO KEEP note of which sessions show incomplete coverage during exploration.
# TODO define maze state at each trial

# TODO finish populating trials table
# TODO make clips

# %%
"""
    Looking at model based V1
    Load data
"""
tloader = TrialsLoader(experiment_name='Model Based', tracking='all')
trials = tloader.load_trials_by_condition()

sessions = list(set((Session  & "experiment_name='Model Based'").fetch('session_name')))
explorations = pd.DataFrame((Explorations * Session * Session.Metadata & "experiment_name='Model Based'").fetch())



# %%
# Plot all explorations and trials
f, axarr = plt.subplots(ncols=6, nrows=4, figsize=(20, 12), sharex=True, sharey=True)

for ax, (i, exp) in zip(axarr.flat, explorations.iterrows()):
    ax.plot(exp.body_tracking[:, 0], exp.body_tracking[:, 1], lw=1, color=[.6, .6, .6])

    trs = trials.loc[trials.session_name == exp.session_name]
    for i, trial in trs.iterrows():
        ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=paper.arms_colors[trial.escape_arm])

    ax.set(title=exp.session_name)


# %%
# Plot p(arm)
ntrials = len(trials)
pleft = len(trials.loc[trials.escape_arm == 'left']) / ntrials
pcenter = len(trials.loc[trials.escape_arm == 'center']) / ntrials
pright = len(trials.loc[trials.escape_arm == 'right']) / ntrials


f, ax = plt.subplots(figsize=(14, 9))
ax.bar([0, 1], [pleft, pcenter + pright], color=[seagreen, salmon])
























# %%
"""
    Looking at model based V2
    Load data
"""
sessions = list(set((Session  & "experiment_name='Model Based V2'").fetch('session_name')))
explorations = pd.DataFrame((Explorations * Session * Session.Metadata & "experiment_name='Model Based V2'").fetch())

stimuli = {s:(Session * Stimuli & f'session_name="{s}"').fetch('overview_frame') for s in sessions}
tloader = TrialsLoader(experiment_name='Model Based', tracking='all')
trials = tloader.load_trials_by_condition()
# import time
# from collections import namedtuple
# from tqdm import tqdm 

# tr = namedtuple('tracking', 'x, y')

# tracking = {}
# for sess in tqdm(sessions):
#     x = (Session * TrackingData.BodyPartData & f'session_name="{sess}"'
#                          & "bpname='body'"  & "experiment_name='Model Based V2'").fetch('x')
#     time.sleep(2)
#     y = (Session * TrackingData.BodyPartData & f'session_name="{sess}"'
#                          & "bpname='body'"  & "experiment_name='Model Based V2'").fetch('y')

#     print(x, y)
#     if len(x) == 0 or len(y) == 0:
#         raise ValueError()

#     time.sleep(2)
#     tracking[sess] = tr(x, y)
     


# %%
# Plot all explorations and trials
fps = 40
nsec = 12


f, axarr = plt.subplots(ncols=6, nrows=4, figsize=(20, 12), sharex=True, sharey=True)

for ax, (i, exp) in zip(axarr.flat, explorations.iterrows()):
    ax.plot(exp.body_tracking[:, 0], exp.body_tracking[:, 1], lw=1, color=[.6, .6, .6])

    sess = exp.session_name
    for stim in stimuli[sess]:
        s, e = stim, stim + (nsec * fps)
        ax.plot(tracking[sess].x[s:e], tracking[sess].y[s:e], 
                            color=paper.arms_colors[trial.escape_arm])

    ax.set(title=sess)


# %%
# Load and plot explorations

explorations = pd.DataFrame((Explorations * Session * Session.Metadata & "experiment_name='Model Based V2'").fetch())
print(explorations)
print(explorations.session_name)

f, axarr = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)

for ax, (i, exp) in zip(axarr.flat, explorations.iterrows()):
    ax.plot(exp.body_tracking[:, 0], exp.body_tracking[:, 1], lw=1, color=[.6, .6, .6])

    ax.set(title=exp.session_name)

# %%
