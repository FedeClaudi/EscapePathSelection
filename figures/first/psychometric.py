# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from myterial import blue_grey_dark, green

from fcutils.plot.elements import ball_and_errorbar
from fcutils.progress import track
from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)
from fcutils.maths.functions import centered_logistic
from fcutils.plot.distributions import plot_fitted_curve

from paper import Tracking

from figures.first import M1, M2, M3, M4, M6
from figures._plot_utils import generate_figure
from figures.colors import tracking_color, start_color, end_color
from figures.statistical_tests import fisher
from fcutils.maths.fit import linear_regression

print(M1, M2, M3, M4, M6, sep='\n\n')



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

# -------------- plot psychometric curve across maze 1->4 (+ 6) -------------- #

# TODO run bayes and cache

ax = generate_figure()

datasets = [M4, M3, M2, M1]
datasets += [M6]

X, Y, YERR = [], [], []
for n, data in enumerate(datasets):
    # Plot global mean for all trials
    prange = data.grouped_pR()
    ball_and_errorbar(data.maze['ratio'], prange.mean, ax, prange=prange, color=data.color, s=100, orientation='vertical', label=data.name)

    if data.name != 'M6':
        X.append(data.maze['ratio'])
        Y.append(prange.mean)
        YERR.append(prange.sem)

    # plt p(R) for each mouse
    ax.scatter(data.maze['ratio'] * np.ones(data.n_mice), data.mice_pR(), s=25, color=data.mice_color, zorder=10)

# plot psychometric
curve_params = plot_fitted_curve(
                centered_logistic,
                X,
                Y,
                ax,
                xrange=[.4, .7], 
                scatter_kwargs=dict(alpha=0),
                fit_kwargs = dict(sigma=YERR),
                line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))

# fix axes
ratios = [data.maze['ratio'] for data in datasets]
ax.legend()
_ = ax.set(title='Psychometric curve', ylabel='$p(R)$', xlabel='maze geodesic ratio', 
            xticks=ratios, xticklabels=[f"{r:.2f}" for r in ratios])


# %%

# ------------------------- plot all trials tracking ------------------------- #

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
# TODO figure out why distance travelled is so weird
# correlation between escape duration and distance travelled
ax = generate_figure()

X, Y, C = [], [], []
for dataset in (M1, M2, M3, M4):
    for i,trial in dataset.iterrows():
        if trial.escape_arm == 'center': continue

        distance_travelled = np.sum(np.sqrt(
            np.diff(trial.x)**2 + np.diff(trial.y)**2
        ))
        X.append(distance_travelled) 
        Y.append(trial.escape_duration)
        C.append(tracking_color if trial.escape_arm=='right' or dataset.name=='M4' else green)

        # ax.plot(trial.x[start:], trial.y[start:])



# plot data points
ax.scatter(X, Y, s=50, c=C, alpha=.6)

# fit linear regression and plot
_, intercept, slope, _ = linear_regression(X, Y, robust=True)

x0, x1 = 80, 250
ax.plot(
    [x0, x1],
    [slope * x0 + intercept, slope*x1 + intercept], 
    lw=2,
    color=[.2, .2, .2]
)

_ = ax.set(title='Distance vs escape duration', ylabel='escape duration\n$s$', xlabel='distance travelled\n$cm$')


# %%
# ---------------------------- alternative hypotheses --------------------------- #

# TODO show that alternative options are not good

# %%

# ---------------------------------------------------------------------------- #
#                                      M6                                      #
# ---------------------------------------------------------------------------- #


table = np.array([
    list(M4.escape_numbers_by_arm().values()),
    list(M6.escape_numbers_by_arm().values())
])

fisher(table, ' p(R) in M4 vs M6')


# get escape duration for each arm in M4 and M6
m4_L = M4.L.escape_duration
m4_R = M4.R.escape_duration

m6_L = M6.L.escape_duration
m6_R = M6.R.escape_duration

ax = generate_figure()
ax.bar(
    [0, 1, 3, 4],
    [m4_L.mean(), m4_R.mean(), m6_L.mean(), m6_R.mean()],
    yerr=[m4_L.sem(), m4_R.sem(), m6_L.sem(), m6_R.sem()],
    color=[M4.color, M4.color, M6.color, M6.color]
)

_ = ax.set(
    title='escape duration by arm',
    ylabel='duration (s)',
    xticks=[0, 1, 3, 4],
    xticklabels=['$M4_L$', '$M4_R$', '$M6_L$', '$M6_R$']
)

# %%
