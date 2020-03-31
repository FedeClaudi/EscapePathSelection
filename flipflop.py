# %%
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt

from fcutils.plotting.colors import *
from fcutils.file_io.io import load_yaml
from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.colors import desaturate_color


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes


arms_colors = {
    "left": teal,
    'center': darkcyan,
    "right": lilla,
}




# %%
# --------------------------------- LOAD DATA -------------------------------- #
# Load trials from both flip flop experiments and concatenate
trials = TrialsLoader(experiment_name = 'FlipFlop2 Maze', tracking='all')

trs1 = trials.load_trials_by_condition(maze_design=None)

trials.experiment_name = 'FlipFlop Maze'
trs2 = trials.load_trials_by_condition(maze_design=None)

trs = pd.concat([trs1, trs2]) # <- trials are here


# --------------------------------- Clean up --------------------------------- #
goodids, skipped = [], 0

_trials = trs
for i, trial in _trials.iterrows():
    if trial.escape_arm == "left":
        if np.max(trial.body_xy[:, 0]) > 600:
            skipped += 1
            continue
    goodids.append(trial.stimulus_uid)

t = trs.loc[trs.stimulus_uid.isin(goodids)]
trs = t

# ---------------------- Load metadata about maze state ---------------------- #
metadata = load_yaml(os.path.join(paths.flip_flop_metadata_dir, "trials_metadata.yml"))
maze_state = []
for i, t in trs.iterrows():
    maze_state.append([v for l in metadata[int(t.uid)]  for k, v in l.items() if k==t.stimulus_uid][0])
trs['maze_state'] = maze_state

left_long_trs = trs.loc[trs.maze_state == 'L']
right_long_trs = trs.loc[trs.maze_state == 'R']

for side, tr in zip(['LEFT', 'RIGHT'], [left_long_trs, right_long_trs]):
    l_esc = len(tr.loc[tr.escape_arm == 'left'])
    r_esc = len(tr.loc[tr.escape_arm == 'right'])
    pr = round(r_esc/len(tr), 3)


# ------------------------------- Compute p(R) ------------------------------- #
baseline_trials = trs.loc[trs.maze_state == 'L']
flipped_trials = trs.loc[trs.maze_state == 'R']

trials.datasets = {'baseline':baseline_trials, 'flipped':flipped_trials}
hits, ntrials, p_r, n_mice, trs = trials.get_binary_trials_per_dataset()
grouped_pRs = trials.grouped_bayes_by_dataset_analytical()

# --------------------------- Print n trials summary -------------------------- #
summary = pd.DataFrame(dict(
    maze=grouped_pRs.dataset.values,
    tot_trials = [len(trials.datasets[m]) for m in grouped_pRs.dataset.values],
    n_mice = list(n_mice.values()),
    avg_n_trials_per_mouse = [np.mean(nt) for nt in list(ntrials.values())]
))
summary




# %%
# --------------------------------- Plotting --------------------------------- #
X_labels = list(grouped_pRs.dataset.values)
X=[.4, .6]
Y = grouped_pRs['mean'].values
Y_err = [sqrt(v)*2 for v in grouped_pRs['sigmasquared']]
colors = [paper.maze_colors['maze1'], desaturate_color(paper.maze_colors['maze1'], k=.4)]



f, ax = create_figure(subplots=False, figsize=(16, 10))

ax.bar(X, Y, width=.2, linewidth=0, yerr=Y_err, color=colors,
            error_kw={'elinewidth':3})

ax.axhline(.5, lw=2, color='k', ls=':', alpha=.5)

ax.set(title="p(R) before and after flip", xlabel='condition', ylabel='p(R)',
        xticks=X, xticklabels=X_labels, xlim=[0.2, .8], ylim=[0, 1],
        )


clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'flipflop_pR'), svg=True)


