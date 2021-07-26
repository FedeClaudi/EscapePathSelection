
# %%
import sys
from pathlib import Path
import pandas as pd
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from scipy.stats import circmean, circstd

import warnings

from fcutils.plot.elements import plot_mean_and_error

from figures._plot_utils import plot_threat_tracking_and_angle, generate_figure
from figures.first import M1, M2, M3, fig_1_path, M4
from figures.settings import trace_downsample, dpi
from figures.colors import tracking_color, tracking_color_dark, start_color, end_color
from figures._data_utils import register_in_time


'''
    Plot tracking on T and average orientaion on T for L vs R
'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

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
import matplotlib.pyplot as plt
from fcutils.plot.figure import clean_axes
# axes = generate_figure(ncols=3, nrows=3, , flatten=False,  gridspec_kw={'width_ratios': [1, 2, 2]})
f = plt.figure(figsize=(19, 14))
# plot average heading while on threat platform
n_samples= 50
for n, dataset in enumerate((M1, M2, M3, M4)):
    axes = [
        plt.subplot(4, 4, 4 * n + 1),
        plt.subplot(4, 4, (4 * n) + 3, projection='polar'),
        plt.subplot(4, 4, (4 * n) + 4, projection='polar'),
        plt.subplot(4, 4, (4 * n) + 2, )
    ]

    _, L, R = plot_threat_tracking_and_angle(dataset, axes=axes, n_samples=n_samples)
    ypos = plot_prediction_accuracy(L, R, axes[3])

    axes[0].set(title=dataset.name)
    axes[0].axis('off')
    axes[1].set_theta_zero_location("N")
    axes[1].set_theta_direction(-1)
    axes[2].set_theta_zero_location("N")
    axes[2].set_theta_direction(-1)
    axes[1].set(title='left trials')
    axes[2].set(title='right trials')
    axes[3].axvline(dataset.pR, lw=2, color=[.2, .2, .2], ls='--')
    axes[3].set(ylabel='Y position', yticks=[],  xlim=[0,1], xlabel='accuracy', xticks=[0, 0.5, 1])

clean_axes(f)
f.tight_layout()
f.savefig(fig_1_path / 'panel_sE_angles_on_T.eps', format='eps', dpi=dpi)



# %%
