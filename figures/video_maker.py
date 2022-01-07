# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('../')

from first import M4, M3, M2, M1
from paper import Stimuli

from fcutils.video import trim_clip, get_video_params

'''
    Make video clips showing escapes on different arms
'''
videos_folder = Path(r'W:\swc\branco\Federico\raw_behaviour\maze\video')
save_folder = Path(r'D:\Dropbox (UCL)\Rotation_vte\Writings\BehavPaper\Clips')
# %%

# ---------------------------------------------------------------------------- #
#                                   M4 TRIAL                                   #
# ---------------------------------------------------------------------------- #

"""
    Makes two clips with L/R escape on each arm of arena 1 
"""

trials = M4.trials.loc[(M4.trials.uid > 88)&(M4.trials.uid < 100)].sort_values('escape_duration')
trials = dict(
    left=(trials.loc[trials.escape_arm == 'left']),  # 1
    right=(trials.loc[trials.escape_arm == 'right']),
    center=(trials.loc[trials.escape_arm == 'center']),

)

for side, trials in trials.items():
    for tnum in range(10):
        # get trial video
        trial = trials.iloc[tnum]

        video = videos_folder / (trial.recording_uid + '.avi')
        assert video.exists()
        
        # get stimulus frame
        stim_frame = (Stimuli & f'stimulus_uid="{trial.stimulus_uid}"').fetch1('overview_frame')

        # make cliip

        trim_clip(
            str(video), save_folder/f'arena1_{side}_{tnum}.mp4', start_frame=stim_frame-30, end_frame=stim_frame + 150
        )

# %%

# ---------------------------------------------------------------------------- #
#                                   PSICHOMETRIC TRIAL                                   #
# ---------------------------------------------------------------------------- #


for dataset in (M1, M2, M3, M4):
    print('Making ', dataset.name)
    
    # if dataset.name == 'M1':
    trials = dataset.trials.loc[dataset.trials.uid > 280].sort_values('escape_duration')
    if dataset.name not in ('M1', 'M2'):
        trials = trials.loc[trials.escape_arm == 'left']

    # else:
    #     trials = dataset.trials.loc[dataset.trials.escape_arm == 'right'].sort_values('escape_duration')
    if len(trials) < 10:
        raise ValueError

    for trialn in range(10):
        trial = trials.iloc[trialn]


        video = videos_folder / (trial.recording_uid + '.avi')
        if not video.exists():
            video = videos_folder / (trial.recording_uid + '_1Overview.mp4')
        assert video.exists(), (dataset.name, video)
        
        # get stimulus frame
        stim_frame = (Stimuli & f'stimulus_uid="{trial.stimulus_uid}"').fetch1('overview_frame')

        # make cliip
        trim_clip(
            str(video), save_folder/f'arena_{5 - int(dataset.name[-1])}_{trialn}.mp4', start_frame=stim_frame-30, end_frame=stim_frame + 150
        )
# %%
# ---------------------------------------------------------------------------- #
#                                   FLIPFLOP TRIAL                                   #
# ---------------------------------------------------------------------------- #
import pandas as pd

from fcutils.path import from_yaml


from paper.helpers.mazes_stats import get_mazes
from paper import Trials, paths
from figures.settings import max_escape_duration
from figures.dataset import DataSet
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

data.maze = get_mazes()['maze1']

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

trials = dict(
    baseline = data.trials.loc[data.trials.maze_state == 'L'].sort_values('escape_duration'),
    flipped = data.trials.loc[data.trials.maze_state == 'R'].sort_values('escape_duration')
)
# %%
for state, trs in trials.items():
    for trialn in range(10):
        trial = trs.iloc[trialn]

        video = videos_folder / (trial.recording_uid + '.avi')
        if not video.exists():
            video = videos_folder / (trial.recording_uid + '_1Overview.mp4')
        assert video.exists(), (dataset.name, video)
        
        # get stimulus frame
        stim_frame = (Stimuli & f'stimulus_uid="{trial.stimulus_uid}"').fetch1('overview_frame')

        # make cliip
        trim_clip(
            str(video), save_folder/f'flipflop_{state}_{trialn}.mp4', start_frame=stim_frame-30, end_frame=stim_frame + 150
        )