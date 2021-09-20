# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.plot.figure import clean_axes
from fcutils.path import from_yaml
from sklearn.model_selection import train_test_split


from paper import Trials, paths
from paper.helpers.mazes_stats import get_mazes
from figures._data_utils import register_in_time

from figures.settings import dpi
from figures.second import fig_2_path
from figures.dataset import DataSet
from figures.colors import M1_color
from figures.settings import max_escape_duration
from figures._plot_utils import plot_threat_tracking_and_angle, generate_figure

'''
    Plot tracking of FlipFlop trials specifically on the threat platform
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

baseline = DataSet('baseline', data.trials.loc[data.trials.maze_state == 'L'])
flipped = DataSet('flipped', data.trials.loc[data.trials.maze_state == 'R'])

# %%

def plot_prediction_accuracy(L, R, ax):
    
    data = pd.concat([L, R]).reset_index()
    X, Y = data.orientation, data.escape_arm
    ypos = np.mean(register_in_time(data.y, n_samples), 1)

    size = 5
    prev_frame_Y = 0
    for frame in np.arange(n_samples):
        frame_Y = ypos[frame]
        if frame_Y - prev_frame_Y > 4:
            prev_frame_Y = frame_Y
                
            scores = []
            for i in range(100):
                xtrain, xtest, ytrain, ytest = train_test_split(X, Y)


                _xtrain = pd.DataFrame(
                    dict(
                        orientation = [np.mean(t[frame:frame+size]) for i,t in xtrain.iteritems()]
                    )
                )

                _xtest = pd.DataFrame(
                    dict(
                        orientation = [np.mean(t[frame:frame+size]) for i,t in xtest.iteritems()]
                    )
                )

                model = LogisticRegression().fit(_xtrain, ytrain)
                scores.append(model.score(_xtest, ytest))

            # ax.bar(frame + size*.5, np.mean(scores), yerr=sem(scores), color=[.5, .5, .5])
            ax.barh(frame_Y, np.mean(scores), height=2.5, xerr=np.std(scores), color=[.5, .5, .5])
    return ypos


# %%
f = plt.figure(figsize=(16, 8))
n_samples= 50

for n, (name, trials) in enumerate(zip(('baseline', 'flipped'), (baseline, flipped))):
    
    axes = [
        plt.subplot(2, 4, 4 * n + 1),
        plt.subplot(2, 4, (4 * n) + 3, projection='polar'),
        plt.subplot(2, 4, (4 * n) + 4, projection='polar'),
        plt.subplot(2, 4, (4 * n) + 2, )
    ]

    _, L, R = plot_threat_tracking_and_angle(trials, axes=axes, n_samples=50)
    ypos = plot_prediction_accuracy(L, R, axes[3])


    axes[0].set(title=trials.name)
    axes[0].axis('off')
    axes[1].set_theta_zero_location("N")
    axes[1].set_theta_direction(-1)
    axes[2].set_theta_zero_location("N")
    axes[2].set_theta_direction(-1)
    axes[1].set(title='left trials')
    axes[2].set(title='right trials')
    axes[3].axvline(trials.pR, lw=2, color=[.2, .2, .2], ls='--')
    axes[3].set(ylabel='Y position', yticks=[],  xlim=[0,1], xlabel='accuracy', xticks=[0, 0.5, 1])

    # break
clean_axes(f)
f.tight_layout()
f.savefig(fig_2_path / 'panel_s1A_angles_on_T.eps', format='eps', dpi=dpi)
# %%
