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

from fcutils.path import from_yaml

from paper import Trials, paths
from paper.helpers.mazes_stats import get_mazes

from figures.dataset import DataSet
from figures.colors import M1_color, tracking_color, start_color, end_color
from figures._plot_utils import generate_figure, plot_trial_tracking
from figures.settings import max_escape_duration
from figures.statistical_tests import fisher
from figures._data_utils import get_number_per_arm_fro_trials_df, get_pR_from_trials_df


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
# plot tracking
axes = generate_figure(ncols=2, figsize=(14, 8), sharex=True, sharey=True)

for ax, trs in zip(axes, (baseline, flipped)):
    for i, trial in trs.iterrows():
        plot_trial_tracking(ax, trial, tracking_color, start_color, end_color)
    ax.axis('off')
axes[0].set(title='baseline')
axes[1].set(title='flipped')
# %%
