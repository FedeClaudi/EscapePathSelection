# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.path import from_yaml
from myterial import salmon, light_blue

from paper import Session, Stimuli, paths, Tracking, Trials


from figures.dataset import DataSet
from figures.colors import tracking_color, start_color, end_color
from figures._plot_utils import generate_figure, plot_trial_tracking


# %%
# get data
notes = from_yaml('mb_annotations.yaml')

# get all trials
sessions = pd.concat([pd.DataFrame(Session & 'experiment_name="Model Based V2"'), 
            pd.DataFrame(Session & 'experiment_name="Model Based V3"')]).reset_index()

stimuli = []
for i, sess in sessions.iterrows():
    stimuli.append(pd.DataFrame(Stimuli & f'uid={sess.uid}'))
    
stimuli = pd.concat(stimuli).reset_index()
print(len(stimuli))


# %%
# select sessions matching criteria
@dataclass
class Selected:
    uid: int
    stimuli: pd.DataFrame
    closed_idx: int
    sequence: list

selected = []
for session in stimuli.uid.unique():
    session_stimuli = stimuli.loc[stimuli.uid == session]

    # get notes
    sequence = []
    for i, stim in session_stimuli.iterrows():
        tag = f'{stim.stimulus_uid}_{stim.overview_frame}'
        sequence.append(notes[tag])

    # stop at skip
    # if 'skip' in sequence:
    #     sequence = sequence[:sequence.index('skip')]

    # get first closed
    if 'closed' not in sequence:
        print('skippy')
        continue
    closed_idx = sequence.index('closed')

    session_stimuli = session_stimuli[:len(sequence)]
    session_stimuli['sequence'] = sequence

    selected.append(
        Selected(
            session, 
            session_stimuli.reset_index(),
            closed_idx,
            sequence
        )
    )

print(f'Selected {len(selected)} sessions out of {len(stimuli.uid.unique())}')

# %%
# get tracking for all sessions 
tracking = []

for session in selected:
    print(session.uid)
    tracking.append(pd.DataFrame(
        Tracking.BodyPart & 'bpname="body"' & f'uid={session.uid}'
    ))
tracking = pd.concat(tracking).reset_index()

# %%

# --------------------------------- plot data -------------------------------- #
# for each session plot exploration
# baseline trials
# closed trial
# after closed trial

def get_trial_arm(x, y):
    '''
        Checks if arm is center (right) or side (left) arm 
        based on tracking
    '''
    try:
        crossed = np.where(y > 100)[0][0]
    except IndexError:
        crossed = len(y)-1

    if x[crossed] < 75 or x[crossed] > 150:
        arm = 0 # side
    else:
        arm = 1
    return arm

closed_color = salmon
after_closed_color = light_blue
n_frames = 40 * 20

excluded_sessions = [  # not enough valid trials or blocked at wrong trial
    '190328_CA512',
    '190412_CA556',
    '190425_CA601',
    '190425_CA602',
    '190531_CA693',
    '190603_CA689',
    '190603_CA681',
    '190604_CA687'
]

axes = generate_figure(ncols=3, nrows=3, figsize=(20, 18), sharex=True, sharey=True)
count = 0
arms = dict(
    baseline = [],
    blocked = [],
    post = [],
)
for session in selected:
    trk = tracking.loc[tracking.uid == session.uid].iloc[0]
    first_stim = session.stimuli.iloc[0].overview_frame

    if session.stimuli.iloc[0].session_name in excluded_sessions:
        continue
    else:
        ax = axes[count]
        count += 1

    for i, stim in session.stimuli.iterrows():
        start = stim.overview_frame
        end = stim.overview_frame + n_frames

        x, y = trk.x[start:end], trk.y[start:end]
        side = get_trial_arm(x, y)
        # if side == 'right':
        #     color = 'red'
        # else:
        #     color = 'k'

        if i < session.closed_idx:
            color = [.2, .2, .2]
            arms['baseline'].append(side)
        elif i == session.closed_idx:
            color = closed_color
            arms['blocked'].append(side)
        elif i == session.closed_idx + 1:
            color = after_closed_color
            arms['post'].append(side)
        elif i == session.closed_idx + 2:
            color = after_closed_color
            arms['post'].append(side)
        else:
            continue

        ax.plot(x, y, color=color, lw=5)


    ax.plot(
        trk.x[40 * 180 : first_stim],
        trk.y[40 * 180 : first_stim],
        color=tracking_color, alpha=.4, zorder=-1
    )
    ax.set(title=stim.session_name)
    ax.axis('off')

    # ax.axvline(75)
    # ax.axvline(150)
    # ax.axhline(100)


for k,v in arms.items():
    print(f'{k} || {len(v)} trials -> p(short) {np.mean(v):.3f}')

# %%
