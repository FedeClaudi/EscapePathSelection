
# %%
# Imports
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt
from collections import namedtuple

from fcutils.plotting.utils import create_figure, clean_axes, save_figure, set_figure_subplots_aspect
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.colors import *
from fcutils.maths.distributions import get_distribution
from fcutils.file_io.io import load_yaml

import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.dbase.TablesDefinitionsV4 import Explorations, Session, Stimuli, TrackingData


savepath = os.path.join(paths.plots_dir, 'modelbased')

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

tloader = TrialsLoader(experiment_name='Model Based V2', tracking='all', escapes_dur = False)
trials = tloader.load_trials_by_condition()

#     print(x, y)
#     if len(x) == 0 or len(y) == 0:
#         raise ValueError()

tracking = pd.read_hdf(os.path.join(paths.cache_dir, 'mbv2tracking.h5'), key='hdf')
tracking.index = tracking.session_name


# %%
# Load metadata
notes = load_yaml('D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\modelbased\\notes.yml')

skips = []

for sess, stims in stimuli.items():
    if sess not in notes.keys():
        print('No notes on session: ', sess, ' ', len(stims), ' stimuli')
        skips.append(sess)
        continue
    trials = notes[sess]['trials']
    if len(trials) != len(stims):
        raise ValueError(f'{sess} - expected {len(trials)} got {len(stims)}')
    else:
        print(f'        {sess} is good')

skips.extend(['190415_CA558', '190426_CA532', '190425_CA602'])

# %%
"""
    Model based V2
    Get tracking afer stimulus and use very rough
    measures to get on which arm the escape was (and if it was an escape)

"""

tracking_mode = {  # tracking is not well registered so need to scale stuff based on it
    '190328_CA503': 's',
    '190328_CA506': 'm',
    '190328_CA508': 'l',
    '190328_CA511': 's',
    '190328_CA512': 'm',
    '190328_CA535': 's',
    '190328_CA536': 'l',
    '190328_CA537': 'm',
    '190412_CA553': 'm',
    '190412_CA554': 'm',
    '190412_CA556': 'm',
    '190412_CA557': 'm',
    '190413_CA555': 'm',
    '190413_CA558': 'm',
    '190413_CA559': 'm',
    '190413_CA600': 'm', 
    '190425_CA531': 'm', 
    '190425_CA601': 'm', 
    '190425_CA602': 'm', 
    '190426_CA532': 'm', 
    '190426_CA557': 'm',
}

bds = namedtuple('bounds', 'xmin, xmax, ymin, ylow')
bounds = {
    's': bds(350, 650, 550, 350),
    'm': bds(300, 700, 750, 450),
    'l': bds(200, 900, 850, 500)
}


f, axarr = plt.subplots(ncols=5, nrows=4, figsize=(20, 12), sharex=False, sharey=False)


fps, nsec = 40, 20
tot, sides = 0, 0

numbers = {
    'baseline': {'tot':0, 'sides':0},
    'block+0': {'tot':0, 'sides':0},
    'block+1': {'tot':0, 'sides':0},
}
colors = {
    'baseline': (salmon, red),
    'block+0': (skyblue, midnightblue),
    'block+1': (lightseagreen, springgreen),
}
toplot ='block+1'

axn = 0
for i, exp in explorations.iterrows():
    sess = exp.session_name
    if sess in skips: continue
    else:
        ax = axarr.flat[axn]
        axn += 1

    ax.plot(exp.body_tracking[:, 0], exp.body_tracking[:, 1], lw=1, color=[.8, .8, .8])

    # get BASELINE trials
    last = np.where(np.array(notes[sess]['trials']) == 0)[0][-1] +1
    baseline = stimuli[sess][:last]

    # get BLOCK trials
    blk = notes[sess]['trials'][last:]
    if last < len(stimuli[sess]):
        block0 = [stimuli[sess][last]]
        block1 = stimuli[sess][last+1:]
    else:
        block0, block1 = [], []


    for condition, condstims in zip(['baseline', 'block+0', 'block+1'], [baseline, block0, block1]):
        for stim in condstims:
            tr = tracking.loc[sess]
            x = tr.x[stim:stim+(fps*nsec)]
            y = tr.y[stim:stim+(fps*nsec)]

            # if np.nanmax(y) < bounds[tracking_mode[sess]].ymin: 
            #             continue # not an escape


            numbers[condition]['tot'] += 1
            try:
                idx = np.where(y >= bounds[tracking_mode[sess]].ymin)[0][0]
            except:
                if condition == 'baseline': continue
                idx = np.argmax(y)

            if y[idx] <=  bounds[tracking_mode[sess]].ylow: continue


            xmin, xmax = bounds[tracking_mode[sess]].xmin, bounds[tracking_mode[sess]].xmax

            if x[idx] < xmin or x[idx] > xmax:
                col = colors[condition][0]
                numbers[condition]['sides'] +=1
            else:
                col = colors[condition][1]


            if condition == toplot:
                ax.plot(x, y, color=col)
                # ax.scatter(x[idx], y[idx], color='r', zorder=99)


    # ax.set(title=sess)
    # ax.axvline( bounds[tracking_mode[sess]].xmin)
    # ax.axvline( bounds[tracking_mode[sess]].xmax)
    # ax.axhline( bounds[tracking_mode[sess]].ymin)
    # ax.axhline( bounds[tracking_mode[sess]].ylow, ls='--')
    ax.axis('off')

for n, (condition, numbs) in enumerate(numbers.items()):
    # print(condition, numbs['tot'],  numbs['sides']/numbs['tot'])
    axarr.flat[-1].bar(n, 1-numbs['sides']/numbs['tot'], color=colors[condition][0])
    axarr.flat[-1].text(n-.25, 1-numbs['sides']/numbs['tot']+0.01, round(1-numbs['sides']/numbs['tot'], 2))


axarr.flat[-1].set(ylabel='p(short)',xticks=[0, 1, 2], xticklabels=['bsl', 'blk+0', 'blk>0'], title='p(short) by condition')

_ = f.suptitle(f'Model based V2 trials: {toplot}')
clean_axes(f)
save_figure(f, os.path.join(savepath, f'MBV2_trials_{toplot}'))
set_figure_subplots_aspect(wspace=.5, hspace=.5)

# %%
# Chi squared
from statsmodels.stats.proportion import proportions_chisquare_allpairs

res = proportions_chisquare_allpairs(np.array([numbers['baseline']['sides'], numbers['block+1']['sides']]), 
                                    np.array([numbers['baseline']['tot'], numbers['block+1']['tot']]))



# %%
"""
Looking at Model Based V3
"""

sessions = list(set((Session  & "experiment_name='Model Based V3'").fetch('session_name')))
explorations = pd.DataFrame((Explorations * Session * Session.Metadata & "experiment_name='Model Based V3'").fetch())
stimuli = {s:(Session * Stimuli & f'session_name="{s}"').fetch('overview_frame') for s in sessions}

tracking = pd.DataFrame((Session * TrackingData * TrackingData.BodyPartData
                    & "bpname='body'").fetch())


# %%
f, axarr = plt.subplots(ncols=5, nrows=2, figsize=(22, 12))


for ax, (i, exp) in zip(axarr.flat, explorations.iterrows()):
    print(exp)
    ax.plot(exp.body_tracking[:, 0], exp.body_tracking[:, 1], lw=1, color=[.8, .8, .8])