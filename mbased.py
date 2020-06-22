
# %%
# Imports
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import hline_to_point, vline_to_point
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.maths.distributions import centered_logistic
from fcutils.plotting.colors import desaturate_color
from fcutils.maths.distributions import get_distribution


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.dbase.TablesDefinitionsV4 import Explorations, Session


# TODO deal with sessions not being registered properly
# TODO KEEP note of which sessions show incomplete coverage during exploration.
# TODO define maze state at each trial

# TODO finish populating trials table
# TODO make clips

# %%
# Load and plot explorations

explorations = pd.DataFrame((Explorations * Session * Session.Metadata & "experiment_name='Model Based V2'").fetch())
print(explorations)
print(explorations.session_name)

f, axarr = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)

for ax, (i, exp) in zip(axarr.flat, explorations.iterrows()):
    ax.plot(exp.body_tracking[:, 0], exp.body_tracking[:, 1], lw=1, color=[.6, .6, .6])

    ax.set(title=exp.session_name)

# %%
