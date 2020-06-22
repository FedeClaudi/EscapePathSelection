# %%
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import namedtuple
from math import sqrt

from fcutils.plotting.colors import *
from fcutils.file_io.io import load_yaml
from fcutils.plotting.utils import create_figure, clean_axes, save_figure, set_figure_subplots_aspect
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.plot_distributions import plot_kde


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.dbase.TablesDefinitionsV4 import Session, TrackingData, Recording, Stimuli


arms_colors = {
    "left": teal,
    'center': darkcyan,
    "right": lilla,
}

save_fld = os.path.join(paths.plots_dir, 'flipflop')


# %%
# --------------------------------- LOAD DATA -------------------------------- #
# Load trials from both flip flop experiments and concatenate
trials = TrialsLoader(experiment_name = 'FlipFlop2 Maze', tracking='all')

trs1 = trials.load_trials_by_condition(maze_design=None)

trials.experiment_name = 'FlipFlop Maze'
trs2 = trials.load_trials_by_condition(maze_design=None)

trs = pd.concat([trs1, trs2]) # <- trials are here


# --------------------------------- Clean up --------------------------------- #
goodids, skipped = [], 0

_trials = trs
for i, trial in _trials.iterrows():
    if trial.escape_arm == "left":
        if np.max(trial.body_xy[:, 0]) > 600:
            skipped += 1
            continue
    goodids.append(trial.stimulus_uid)

t = trs.loc[trs.stimulus_uid.isin(goodids)]
trs = t

# ------------------------------ Drop wrong data ----------------------------- #
trs = trs.drop(trs.loc[trs.uid == 183].index)
trs = trs.drop(trs.loc[trs.session_name == '181107_CA344.2'].index)

# ---------------------- Load metadata about maze state ---------------------- #

metadata = load_yaml(os.path.join(paths.flip_flop_metadata_dir, "trials_metadata.yml"))
maze_state = []
for i, t in trs.iterrows():
    maze_state.append([v for l in metadata[int(t.uid)]  for k, v in l.items() if k==t.stimulus_uid][0])
trs['maze_state'] = maze_state

left_long_trs = trs.loc[trs.maze_state == 'L']
right_long_trs = trs.loc[trs.maze_state == 'R']

for side, tr in zip(['LEFT', 'RIGHT'], [left_long_trs, right_long_trs]):
    l_esc = len(tr.loc[tr.escape_arm == 'left'])
    r_esc = len(tr.loc[tr.escape_arm == 'right'])
    pr = round(r_esc/len(tr), 3)


# ------------------------------- Compute p(R) ------------------------------- #
baseline_trials = trs.loc[trs.maze_state == 'L']
flipped_trials = trs.loc[trs.maze_state == 'R']

trials.datasets = {'baseline':baseline_trials, 'flipped':flipped_trials}
hits, ntrials, p_r, n_mice, trs = trials.get_binary_trials_per_dataset()
grouped_pRs = trials.grouped_bayes_by_dataset_analytical()

# --------------------------- Print n trials summary -------------------------- #
summary = pd.DataFrame(dict(
    maze=grouped_pRs.dataset.values,
    tot_trials = [len(trials.datasets[m]) for m in grouped_pRs.dataset.values],
    n_mice = list(n_mice.values()),
    avg_n_trials_per_mouse = [np.mean(nt) for nt in list(ntrials.values())]
))
summary


"""
    Get whole session tracking for each session
"""
sessions = baseline_trials.session_name.unique()

trk = namedtuple('traking', 'x, y, s')
rec_lengths, body_tracking = {}, {}
for sess in sessions:
    x = (Session * TrackingData * TrackingData.BodyPartData \
                    & f'session_name="{sess}"' & 'bpname="body"').fetch('x')
    y = (Session * TrackingData * TrackingData.BodyPartData \
                    & f'session_name="{sess}"' & 'bpname="body"').fetch('y')
    s = (Session * TrackingData * TrackingData.BodyPartData \
                    & f'session_name="{sess}"' & 'bpname="body"').fetch('speed')

    speed = np.hstack(s)
    speed[speed > np.percentile(speed, 90)] = np.percentile(speed, 90)

    body_tracking[sess] = trk(np.hstack(x), np.hstack(y), speed)
    rec_lengths[sess] = [len(xx) for xx in x]





# %%

"""
    Plot overall baseline vs flipped p(R)
"""

X_labels = list(grouped_pRs.dataset.values)
X=[.4, .6]
Y = grouped_pRs['mean'].values
Y_err = [sqrt(v)*2 for v in grouped_pRs['sigmasquared']]
colors = [paper.maze_colors['maze1'], desaturate_color(paper.maze_colors['maze1'], k=.4)]



f, ax = create_figure(subplots=False, figsize=(16, 10))

ax.bar(X, Y, width=.2, linewidth=0, yerr=Y_err, color=colors,
            error_kw={'elinewidth':3})

ax.axhline(.5, lw=2, color='k', ls=':', alpha=.5)

ax.set(title="p(R) before and after flip", xlabel='condition', ylabel='p(R)',
        xticks=X, xticklabels=X_labels, xlim=[0.2, .8], ylim=[0, 1],
        )


clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'flipflop_pR'), svg=True)





# %%
"""
    Plot explorations and flipped explorations 

    # TODO get exploration tracking + flipped expl tracking
    # TODO plot individual trials
    # TODO p(R) first trial of each session + first flipped trials
"""



# %%
"""
    Plot all explorations
"""
def get_stim_comulative_frame_num_in_sess(stim, rec_uid, rlengths):
        """
            stim should be a pandas series with
            necessary metadata

            rec_uid should be a list of strings with UIDs
            of recordings belonging to the session the stim is in

            rlengths should be a list of integers with length of each recording
        """

        first_stim_rec = rec_uid.index(stim.recording_uid)
        pre_stim_frames = np.sum([x for i, x in enumerate(rlengths) if i<first_stim_rec])
        
        try:
            return int(pre_stim_frames + stim.overview_frame)
        except:
            return int(pre_stim_frames + stim.stim_frame)


flipped_expl_shifts = {
    '181114_CA3151':220,
    '191216_CA831': 25,
    '191217_CA833': 480,
    '180928_CA3164': 90,
    '180929_CA3273': 80,
    '180929_CA3274': 165,
    '180929_CA3275': 100,

}




f, axarr = plt.subplots(ncols=6, nrows=4, figsize=(22, 12))
f.suptitle('Flip Flop explorations')
for ax in axarr.flat:    ax.axis('off')


n = 0
duration, distance = {'baseline':[], 'flipped':[]}, {'baseline':[], 'flipped':[]}

for sess in sessions:
    # Get first trial
    baseline = baseline_trials.loc[baseline_trials.session_name == sess]
    flipped = flipped_trials.loc[flipped_trials.session_name == sess]

    if not len(baseline) or not len(flipped): 

        continue

    # Get stimuli times
    bs, fp = baseline.iloc[0], flipped.iloc[0]

    stimuli = pd.DataFrame((Session * Stimuli & f'session_name="{sess}"').fetch())
    stimuli = stimuli.loc[stimuli.overview_frame > 0]
    if not len(stimuli): raise ValueError

    recs = list((Session * Recording 
                                & f'session_name="{sess}"').fetch('recording_uid'))
    
    # Get frame for first baseline trial
    bsframe = get_stim_comulative_frame_num_in_sess(
                        stimuli.iloc[0], recs, rec_lengths[sess])

    if bsframe > len(body_tracking[sess].x): 
        print(f'Might be missing some tracking for {sess}')
        continue
        # raise ValueError

    # get frame for last baseline and first flip trials
    metadata_stims = [(k, v) for s in metadata[fp.uid] for k,v in s.items()]
    blast = [n for n,c in metadata_stims if c == 'L'][-1]
    first = [n for n,c in metadata_stims if c == 'R'][0]

    try:
        blast = baseline_trials.loc[baseline_trials.stimulus_uid == blast].iloc[0]
        dur = (blast.at_shelter_frame - blast.stim_frame)
    except:
        blast = stimuli.loc[stimuli.stimulus_uid == blast].iloc[0]
        dur = 30 * bs.fps


    bsend = get_stim_comulative_frame_num_in_sess(
        blast, 
        recs, rec_lengths[sess]
    ) + dur

    if sess in flipped_expl_shifts.keys():
        bsend += flipped_expl_shifts[sess] * bs.fps

    fpstart = get_stim_comulative_frame_num_in_sess(
        stimuli.loc[stimuli.stimulus_uid == first].iloc[0], 
        recs, rec_lengths[sess]
    )

    if bsend >= fpstart: raise ValueError


    # Get duration and distance covered for each condition
    duration['baseline'].append((bsframe/bs.fps)/60)
    duration['flipped'].append(((fpstart-bsend)/bs.fps/60))

    distance['baseline'].append(np.nansum(body_tracking[sess].s[int(60*bs.fps):bsframe]))
    distance['flipped'].append(np.nansum(body_tracking[sess].s[bsend:fpstart]))

    # Plot
    ax = axarr.flat[n]
    ax.plot(body_tracking[sess].x[300:bsframe], body_tracking[sess].y[300:bsframe], 
                            color=[.6, .6, .6])
    ax.plot(bs.body_xy[:, 0], bs.body_xy[:, 1], color='salmon')
    ax.plot(body_tracking[sess].x[bsend:fpstart]+550, body_tracking[sess].y[bsend:fpstart], 
                            color=[.3, .3, .3])
    ax.plot(fp.body_xy[:, 0]+550, fp.body_xy[:, 1], color='salmon')
    ax.set(title=f'{sess}\n ~{round(duration["baseline"][-1])}min        ~{round(duration["flipped"][-1])}min')


    n += 1



# bins = axarr.flat[-2].hist(duration['baseline'], color='green',  bins=15,
#                 density=True, alpha=.5, label='baseline')
# axarr.flat[-2].hist(duration['flipped'], color='red', bins=bins[1], 
#                 density=True, alpha=.5, label='flipped')

# bins = axarr.flat[-1].hist(distance['baseline'], color='green',  bins=15,
#                 density=True, alpha=.5, label='baseline')
# axarr.flat[-1].hist(distance['flipped'], color='red', bins=bins[1],
#                 density=True, alpha=.5, label='flipped')


plot_kde(data = duration['baseline'], ax = axarr.flat[-2], 
                kde_kwargs=dict(bw=2),
                color=[.6, .6, .6], alpha=.5, label='baseline')
plot_kde(data = duration['flipped'], ax = axarr.flat[-2], 
                kde_kwargs=dict(bw=2),
                color=[.2, .2, .2], alpha=.3, label='flipped')

plot_kde(data = distance['baseline'], ax = axarr.flat[-1], 
                kde_kwargs=dict(bw=3500),
                color=[.6, .6, .6], alpha=.5, label='baseline')
plot_kde(data = distance['flipped'], ax = axarr.flat[-1], 
                kde_kwargs=dict(bw=3500),
                color=[.2, .2, .2], alpha=.3, label='flipped')


axarr.flat[-1].axis('on')
axarr.flat[-1].set(title='Distance covered', xlabel='px')
axarr.flat[-1].legend()

axarr.flat[-2].axis('on')
axarr.flat[-2].set(title='Duration', xlabel='min')
_ = axarr.flat[-2].legend()

clean_axes(f)
set_figure_subplots_aspect(wspace=0.5, hspace=0.3)
save_figure(f, os.path.join(save_fld, 'ff_exploration'))



# %%
"""
    Plot p(R) of each first trial (baseline vs flipped)
"""


baselines, flippeds = [], []
f, axarr = plt.subplots(ncols=2, figsize=(22, 12))

for sess in sessions:
    baseline = baseline_trials.loc[baseline_trials.session_name == sess]
    flipped = flipped_trials.loc[flipped_trials.session_name == sess]

    if len(baseline) and len(flipped):
        bs, fp = baseline.iloc[0], flipped.iloc[0]
        if bs.escape_arm == 'left':
            baselines.append(0)
        else:
            baselines.append(1)

        if fp.escape_arm == 'left':
            flippeds.append(0)
        else:
            flippeds.append(1)


        axarr[0].plot(bs.body_xy[:, 0], bs.body_xy[:, 1], c=arms_colors[bs.escape_arm])
        axarr[1].plot(fp.body_xy[:, 0], fp.body_xy[:, 1], c=arms_colors[fp.escape_arm])


print(f'First trial p(R) baseline {round(np.mean(baselines), 2)} - p(L) flipped {round(1 - np.mean(flippeds), 2)} - n mice: {len(flippeds)}')


# %%
