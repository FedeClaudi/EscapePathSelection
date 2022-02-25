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
from figures.first import fig_1_path

# %%

# get sessions
sessions = dict(
    deadend=pd.DataFrame(Session().get_by(maze=4)),
    open=pd.DataFrame((Session & 'experiment_name="Revision1"').fetch())
)


other_deadend = pd.DataFrame((Session & 'experiment_name="TwoAndAhalf Maze"' & 'uid>1000').fetch())
sessions['deadend'] = pd.concat([sessions['deadend'], other_deadend])

sessions['open'] = sessions['open'].loc[sessions['open']['date'].apply(str) == '2022-01-18']

# get stimuli
stimuli = dict()
for name, sess in sessions.items():
    exp_stimuli = Stimuli.get_by_sessions(sess)
    if name == 'deadend':
        exp_stimuli = exp_stimuli.loc[(exp_stimuli.uid > 88)&(exp_stimuli.uid < 100)]
    stimuli[name] = exp_stimuli

print(len(sessions['deadend']), len(sessions['open']))

print(len(stimuli['deadend']), len(stimuli['open']))

# %%
axes = generate_figure(ncols=2, figsize=(16, 9))
axes = dict(open=axes[1], deadend=axes[0])

ToT =  dict(open={'left': [], 'right': [], 'center':[]}, deadend={'left': [], 'right': [], 'center':[]})
center, left, right =  dict(open=0, deadend=0), dict(open=0, deadend=0), dict(open=0, deadend=0)
mice  = dict(open=[], deadend=[])

for name, stims in stimuli.items():
    for i, stim in track(stims.iterrows(), description='Organizing data', total=len(stims)):
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

            if tracking[out, -1] <= 20:
                # speed below mean - 2*std: ignoring it
                continue

            if out > 4 * 40:
                # too long after stim onset
                continue
            
            # get which platform the mouse steps onto
            x_out = tracking[out, 0]
            if x_out < 105:
                color = 'red'
                left[name] += 1
                ToT[name]['left'].append(out/40)
            elif x_out > 115:
                right[name] += 1
                color='blue'
                ToT[name]['right'].append(out/40)
            else:
                color = 'green'
                center[name] += 1
                ToT[name]['center'].append(out/40)
            mice[name].append(stim.mouse_id)

            axes[name].plot(tracking[:, 0], tracking[:, 1], color=color)
            axes[name].scatter(tracking[out, 0], tracking[out, 1], color='k', s=50, zorder=10)

# %%
from scipy.stats import binom_test


# center['deadend'] = center['deadend'] + 7
# left['deadend'] = left['deadend'] + 25
tots = dict()
for name in ('deadend', 'open'):
    l, c, r = left[name], center[name], right[name]
    tot = l + c + r
    tots[name] = tot

    print(f'{name} N trials {tot} | n mice {len(set(mice[name]))}')
    for arm, n in zip(('side', 'center'), (l+r, c)):
        print(f'p({arm}): {n/tot:.2f}')


    res = binom_test(c, tot, p=0.33) # , alternative='less')
    print(res)


# %%
from statsmodels.stats.proportion import proportions_ztest

counts = [center['deadend'], center['open']]
nobs = [tots['deadend'], tots['open']]
stat, pval = proportions_ztest(counts, nobs)

print(pval)

# %%
# # plot speed at ToT for all trials
# speeds , tots= [], []
# ax = generate_figure()
# for dataset in (M1, M2, M3, M4, M6):
#     for i, trial in dataset.iterrows():
#         speeds.append(trial.speed[trial.out_of_t_frame - trial.stim_frame])
#         tots.append(trial.time_out_of_t)

# ax.hist(tots, bins=20)
# # %%
# mu = np.mean(speeds)
# sdev = np.std(speeds)

# print(f'Mean speed: {mu:.2f}, +- {sdev:.2f} - thershold: {mu -sdev:.2f}')
# # %%
# print(f'Mean ToT: ', {arm: f'{np.mean(v):.2f} +/- {np.std(v):.2f}' for arm,v in ToT.items()})
# # %%
# from scipy.stats import chisquare

# observed = [left, center, right]
# expected = [0.33 * tot, 0.33 * tot, 0.33 * tot]
# chisquare(observed, expected)


# %%
