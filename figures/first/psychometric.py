# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from myterial import blue_grey_dark

from fcutils.progress import track
from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)

from paper.helpers.mazes_stats import get_mazes
from paper import Tracking

from figures.first import M1, M2, M3, M4, M6
from figures._plot_utils import generate_figure
from figures.colors import tracking_color, start_color, end_color
from figures.statistical_tests import fisher
from fcutils.maths.fit import linear_regression

# print(M1, M2, M3, M4, M6, sep='\n\n')
mazes = get_mazes()


# %% get

# ---------------------------------------------------------------------------- #
#                                   M1 TO M4                                   #
# ---------------------------------------------------------------------------- #

# for M1 and M4 check if p(R) != 0.5
print(f'''
    Escape probabilities by arm:
        M1 {M1.escape_probability_by_arm()}
        M2 {M2.escape_probability_by_arm()}
        M3 {M3.escape_probability_by_arm()}
        M4 {M4.escape_probability_by_arm()}
        M6 {M6.escape_probability_by_arm()}
''')

table = np.array([
    list(M1.escape_numbers_by_arm().values()),
    list(M4.escape_numbers_by_arm().values())
])


fisher(table, ' p(R) in M4 vs M1')

# %%
# plot psychometric curve across maze 1->4
# TODO run bayes and cache

# TODO fit psychometric curve
# TODO check if/when same mouse was used across mazes and look at p(R)
# TODO add len(right path)/max len right path

ax = generate_figure()

for n, data in enumerate((M4, M3, M2, M1)):
    # Plot global mean for all trials
    ax.scatter(n, data.pR, s=50, color=data.color, lw=1, edgecolors=blue_grey_dark, zorder=100)

    # plt p(R) for each mouse
    ax.scatter(n * np.ones(data.n_mice), data.mice_pR(), s=25, color=data.mice_color, zorder=10)

# %%
# plot all trials tracking
axes = generate_figure(ncols=3, nrows=2, figsize=(22, 12), sharex=True, sharey=True)

for ax, dataset in track(zip(axes, (M1, M2, M3, M4, M6)), total=5):
    for i, trial in dataset.iterrows():
        ax.plot(trial.x, trial.y, lw=1, alpha=1, color=tracking_color)
        ax.scatter(trial.x[0], trial.y[0], s=30, color=start_color, zorder=10)
        ax.scatter(trial.x[-1], trial.y[-1], s=30, color=end_color, zorder=10)

    ax.set(title=dataset.name)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
axes[-1].axis('off')

# %%
# ------------------------------- supplementary ------------------------------ #

# TODO correlation between escape duration and path length for all trials
# TODO figure out why M6 trials have such long distances
ax = generate_figure()

X, Y = [], []
for dataset in (M1, M2, M3, M4, M6):
    if dataset.name != 'M6': continue
    for i,trial in dataset.iterrows():
        distance_travelled = np.sum(get_speed_from_xy(trial.x, trial.y))
        X.append(distance_travelled)  # because I didnt compute it correctly
        Y.append(trial.escape_duration)


# plot data points
ax.scatter(X, Y, s=50, color=tracking_color, alpha=.6)

# fit linear regression and plot
_, intercept, slope, _ = linear_regression(X, Y)

x0, x1 = 40, 90
ax.plot(
    [x0, x1],
    [slope * x0 + intercept, slope*x1 + intercept], 
    lw=2,
    color=[.2, .2, .2]
)

_ = ax.set(title='Distance vs escape duration', ylabel='escape duration', xlabel='distance travelled')

# TODO show that alternative options are not good

# %%

# ---------------------------------------------------------------------------- #
#                                      M6                                      #
# ---------------------------------------------------------------------------- #

# TODO check p(R) in M6 and if its different from M4
# TODO show that in M4 escape duration is the same for both paths
