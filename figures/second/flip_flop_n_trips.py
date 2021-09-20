# %%
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.path import from_yaml

from paper import Trials, paths, Tracking
from paper.helpers.mazes_stats import get_mazes


from figures.dataset import DataSet
from figures.colors import M1_color
from figures.settings import max_escape_duration

'''
    Get number of shelter/threat trips during baseline 
    and flipped epxloration
'''

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


# %%
# %%
def get_trips(rois, source, target, tracking):
    '''
        Given a list of roi indices for where the mouse is at for every frame, 
        it gets all the times that the mouse did a trip source -> target
    '''
    in_target = np.zeros(len(rois))
    in_target[rois == target] = 1

    at_target = np.where(np.diff(in_target)>0)[0]

    trips = []
    for tgt in at_target:
        # go back in time and get the last time the animal was at source
        try:
            at_source = np.where(rois[:tgt] == source)[0][-1]
        except  IndexError:
            continue

        # if the mouse gets to tgt any other time during the trip, ignore
        if np.any(rois[at_source:tgt-2] == target):
            continue

        if tgt - at_source < 50: 
            # too fast
            continue

        # get arm
        if np.min(tracking[at_source: tgt, 0]) <= 400:
            arm = 'left'
        else:
            arm = 'right'

        # check if there was an error
        # if tracking[tgt, 0] < 200 or tracking[tgt, 0] > 800:
        #     continue

        # get distance travelled
        # x = tracking[at_source:tgt, 0]
        # y = tracking[at_source:tgt, 1]
        # dist = get_dist(x, y)
        dist = np.sum(tracking[at_source:tgt, 2]) * 0.22
        if dist < 25:
            continue  # too short

        trips.append((at_source, tgt, arm, dist))
    return trips

# %%%%%
# ---------------------------------------------------------------------------- #
#                              Get number of trips                             #
# ---------------------------------------------------------------------------- #

delay = {
    105: 100,
    109: 900,
    471:500,
}

n_trips_flipped = []
for sess in baseline.uid.unique():
    if sess == 463 or sess == 107:
        continue  # bad tracking
    try:
        session_delay = delay[sess]
    except KeyError:
        session_delay = 9


    # if sess not in delay.keys():
    #     continue

    # last baseline trials, first flipped
    try:
        bs = baseline.loc[baseline.uid == sess].iloc[-1].stim_frame_session + 40 * session_delay
        flp = flipped.loc[flipped.uid == sess].iloc[0].stim_frame_session
    except IndexError:
        continue
    if flp <= bs: 
        raise ValueError('Nein')

    # Get tracking after flipping
    tracking = pd.DataFrame((Tracking * Tracking.BodyPart & 'bpname="body"' & f'uid={sess}').fetch()).iloc[0]

    roi = tracking.roi[bs:flp]
    body_tracking = np.vstack(tracking[['x', 'y', 'speed']]).T[bs:flp]

    # get trips
    trips_to_shelt = get_trips(roi, 1, 0, body_tracking)
    trips_to_threat = get_trips(roi, 0, 1, body_tracking)
    n_trips_flipped.append(len(trips_to_shelt+trips_to_threat))

    # plot em
    import matplotlib.pyplot as plt
    f, axes = plt.subplots(ncols=2)
    f.suptitle(f'{sess} ({session_delay}s)')

    for trip in trips_to_shelt:
        axes[0].plot(body_tracking[trip[0]:trip[1], 0], body_tracking[trip[0]:trip[1], 1], color='k')
        axes[0].scatter(body_tracking[trip[0], 0], body_tracking[trip[0], 1], color='r', zorder=100, s=100)
        axes[0].scatter(body_tracking[trip[1], 0], body_tracking[trip[1], 1], color='g', zorder=100, s=100)

    for trip in trips_to_threat:
        axes[1].plot(body_tracking[trip[0]:trip[1], 0], body_tracking[trip[0]:trip[1], 1], color='k')
        axes[1].scatter(body_tracking[trip[0], 0], body_tracking[trip[0], 1], color='r', zorder=100, s=100)
        axes[1].scatter(body_tracking[trip[1], 0], body_tracking[trip[1], 1], color='g', zorder=100, s=100)

print(f'Number of trips after flip: {np.mean(n_trips_flipped):.3f} +/- {np.std(n_trips_flipped):.3f}')

# %%
