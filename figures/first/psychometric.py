# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.progress import track

from figures.first import M1, M2, M3, M4, M6
from figures._plot_utils import generate_figure
from figures.colors import tracking_color, start_color, end_color

from figures.statistical_tests import fisher

print(M1, M2, M3, M4, M6, sep='\n\n')
# TODO remove trials not terminating at shelter

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
# TODO add p(R) for each mouse method to dataset
# TODO get colors for each dataset?
# TODO fit psychometric curve
# TODO check if/when same mouse was used across mazes and look at p(R)
# TODO add len(right path)/max len right path

ax = generate_figure()

for n, data in enumerate((M4, M3, M2, M1)):
    ax.scatter(n, data.pR)

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

colors = 'rgbmk'
for color, dataset in zip(colors, (M1, M2, M3, M4, M6)):
    for i,trial in dataset.iterrows():
        # calc distance travelled
        dist = np.sum(trial.speed)

        ax.scatter(dist, trial.escape_duration, s=10, color=color)
ax.set(xlim=[6000, 10000])
# TODO show that alternative options are not good

# %%

# ---------------------------------------------------------------------------- #
#                                      M6                                      #
# ---------------------------------------------------------------------------- #

# TODO check p(R) in M6 and if its different from M4
# TODO show that in M4 escape duration is the same for both paths
