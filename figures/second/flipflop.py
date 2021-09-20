# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.plot.distributions import plot_distribution
from fcutils.path import from_yaml

from paper import Trials, paths
from paper.helpers.mazes_stats import get_mazes

from figures.settings import dpi
from figures.second import fig_2_path
from figures.dataset import DataSet
from figures.colors import M1_color, tracking_color, start_color, end_color
from figures._plot_utils import generate_figure, plot_trial_tracking, triple_plot
from figures.settings import max_escape_duration
from figures.statistical_tests import fisher
from figures._data_utils import get_number_per_arm_fro_trials_df, get_pR_from_trials_df
from figures.bayes import Bayes


# %%
# load data
t1 = Trials.get_by_condition(experiment_name='FlipFlop Maze', escape_duration=max_escape_duration)
t2 = Trials.get_by_condition(experiment_name='FlipFlop2 Maze', escape_duration=max_escape_duration)

trs = pd.concat([t1, t2]).reset_index()
trs = trs.drop(trs.loc[trs.uid == 183].index)
trs = trs.drop(trs.loc[trs.session_name == '181107_CA344.2'].index)

data = DataSet(
    'FlipFlop',
    trs
)
data.color = M1_color
data.maze = get_mazes()['maze1']

print(data)
# %%
# get metadata
metadata = from_yaml(os.path.join(paths.flip_flop_metadata_dir, "trials_metadata.yml"))

maze_state, to_drop = [], []
for i, trial in data.iterrows():
    try:
        session_meta = {k: v for d in metadata[trial.uid] for k, v in d.items()}
        maze_state.append(session_meta[trial.stimulus_uid])
    except KeyError:
        to_drop.append(i)
        maze_state.append('-1')
    
data.trials['maze_state'] = maze_state
_ = data.trials.drop(to_drop)

print(data)

baseline = data.trials.loc[data.trials.maze_state == 'L']
flipped = data.trials.loc[data.trials.maze_state == 'R']

print(f'{len(baseline)}  baseline trials and {len(flipped)} flipped')

# do fisher's test
table = np.array([
    get_number_per_arm_fro_trials_df(baseline),
    get_number_per_arm_fro_trials_df(flipped),
])

print(f'''
        baseline: {get_pR_from_trials_df(baseline):.3f},
        flipped: {get_pR_from_trials_df(flipped):.3f}
    ''')
fisher(table, ' p(R) in baseline vs flipped')

# %%
# get average second exploration duration
exploration_durations = []
for sess in baseline.uid.unique():
    # last baseline trials
    try:
        bs = baseline.loc[baseline.uid == sess].iloc[-1].stim_frame_session / 40 / 60
        flp = flipped.loc[flipped.uid == sess].iloc[0].stim_frame_session / 40 / 60
    except IndexError:
        continue
    exploration_durations.append(flp - bs)

print(f'Second exploration duration: {np.median(exploration_durations):.2f} +/- {np.std(exploration_durations):.2f}')

# %%
# plot tracking
axes = generate_figure(ncols=2, figsize=(14, 8), sharex=True, sharey=True)

for ax, trs in zip(axes, (baseline, flipped)):
    for i, trial in trs.iterrows():
        plot_trial_tracking(ax, trial, tracking_color, start_color, end_color)
    ax.axis('off')
axes[0].set(title='baseline')
axes[1].set(title='flipped')

axes[1].figure.savefig(fig_2_path / 'flip_flop_trackibg.eps', format='eps', dpi=dpi)

# %%
# plot posteriors with box plot
ax = generate_figure()
bayes = Bayes()
for n, (trials, color, name) in enumerate(zip((baseline, flipped), 'kr', ('baseline', 'flipped'))):
    n_trials = len(trials)
    nR = len(trials.loc[trials.escape_arm == 'right'])

    mice_pr = []
    for mouse in trials.mouse_id.unique():
        trs = trials.loc[trials.mouse_id == mouse]
        mice_pr.append(len(trs.loc[trs.escape_arm == 'right'])/len(trs))

    a, b, _, _, _, _, _ = bayes.grouped_bayes_analytical(n_trials, nR)
    plot_distribution(a, b, dist_type='beta', ax=ax, plot_kwargs=dict(color=color, label=name), shaded=True)

    triple_plot(
        -n*2 - 2, 
        mice_pr,
        ax, 
        kde_kwargs=dict(bw=0.05),
        kde_normto=.4,
        box_width=.2,
        color=color,
        fill=.001,
        horizontal=True,
        show_kde=False,
        scatter_kws=dict(s=30),
        spread=0.3)

ax.axhline(0, lw=2, color='k')
ax.plot([0.5, 0.5], [0, 9], ls='--', lw=2, color=[.4, .4, .4], zorder=-1)
ax.legend()
ax.set(ylabel='density', xlabel='p(R)', xlim=[-0.02, 1.02])
ax.figure.savefig(fig_2_path / 'flip_flop_posteriors.eps', format='eps', dpi=dpi)

# %%
# Get out of threat time and on threat platform dynamics
from figures._plot_utils import plot_threat_tracking_and_angle
from figures.dataset import DataSet
print(f'Time out of T: baseline {baseline.time_out_of_t.mean():.2f} +- {baseline.time_out_of_t.std():.2f}, flipped {flipped.time_out_of_t.mean():.2f} +- {flipped.time_out_of_t.std():.2f}')

f = plt.figure(figsize=(16, 9))

#  %%
from scipy.stats import ttest_ind as ttest

'''
Run a ttest to check for independence of ToT duration 
'''

ttest(baseline.time_out_of_t, flipped.time_out_of_t, equal_var=False)



# %%
print(*list(baseline.time_out_of_t), sep='\n')
# %%
print(*list(flipped.time_out_of_t), sep='\n')

# %%
bl = baseline.groupby('session_name').mean().time_out_of_t
fp = flipped.groupby('session_name').mean().time_out_of_t


data = {k: (bl[k], fp[k]) for k in bl.index if k in fp.index}
ttest([d[0] for d in data.values()], [d[1] for d in data.values()], equal_var=True)

# %%
plt.scatter(np.zeros(len(data)), [d[0] for d in data.values()], s=100, zorder=100)
plt.scatter(np.ones(len(data)), [d[1] for d in data.values()], s=100, zorder=100)
plt.plot(np.array(list(data.values())).T, color='k', zorder=1)