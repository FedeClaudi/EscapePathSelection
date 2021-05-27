
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

import warnings

from figures._plot_utils import plot_threat_tracking_and_angle, generate_figure
from figures.first import M1, M2, M3, fig_1_path
from figures.settings import dpi


'''
    Plot tracking on T and average orientaion on T for L vs R
'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

# %%

def plot_prediction_accuracy(L, R, ax):
    
    data = pd.concat([L, R]).reset_index()
    X, Y = data.orientation, data.escape_arm

    frame = 0
    size = 5

    while frame < n_samples:
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

        ax.bar(frame + size*.5, np.mean(scores), yerr=sem(scores), color=[.5, .5, .5])
        frame += size

# %%
axes = generate_figure(ncols=3, nrows=3, figsize=(19, 12), flatten=False,  gridspec_kw={'width_ratios': [1, 2, 2]})

# plot average heading while on threat platform
n_samples= 50
for n, dataset in enumerate((M1, M2, M3)):
    _, L, R = plot_threat_tracking_and_angle(dataset, axes=axes[n, :2], n_samples=n_samples)
    plot_prediction_accuracy(L, R, axes[n, -1])
    axes[n, -1].axhline(dataset.pR, lw=2, color=[.2, .2, .2], ls='--')
    axes[n, -1].set(ylabel='predicted p(R)', xticks=[0, n_samples], xticklabels=['0', '1'], xlabel='normalized time on T')

axes[0, 0].figure.savefig(fig_1_path / 'panel_sE_angles_on_T.eps', format='eps', dpi=dpi)

# %%
