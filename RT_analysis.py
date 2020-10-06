# %%
import numpy as np
import os
import matplotlib.pyplot as plt

import pandas as pd
from fcutils.plotting.utils import clean_axes, save_figure
from fcutils.plotting.colors import desaturate_color
import pyinspect as pi

import paper
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper import paths
# %%
# --------------------------------- Load data -------------------------------- #
_mazes = get_mazes()

print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'all',
    escapes_dur = True,
)

trials = TrialsLoader(**params)
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials
trials.keep_catwalk_only()
# %%
"""
Look at distribution of trials by RT

"""
f, axarr = plt.subplots(nrows=5, sharex=True, figsize=(16, 9))
f2, ax2 = plt.subplots(figsize=(8, 12))

bins=np.linspace(0, 7, 25)
all_tots = np.concatenate([t.time_out_of_t for t in trials.datasets.values()])
for ax in axarr:
    ax.hist(all_tots, bins=bins, color=[.8, .8, .8], zorder=-1, density=True)

mazes = list(trials.datasets.keys())
for n, (maze, trs) in enumerate(trials.datasets.items()):

    tot = trs.time_out_of_t

    axarr[n].hist(tot, bins=bins, label=maze, color=paper.maze_colors[maze], 
            alpha=.6, zorder=99, lw=2, density=True)

    axarr[n].axvline(np.median(tot), lw=8, color=[.4, .4, .4], zorder=101)

    axarr[n].set(xlabel='ToT (s)', ylabel=f'{maze}\nDensity')

    # Split trials based on group median and look at PR
    fast = trs.loc[trs.time_out_of_t < np.median(tot)]
    slow = trs.loc[trs.time_out_of_t > np.median(tot)]

    all_pr = np.mean([1 if e == 'right' else 0 for e in trs.escape_arm])
    fast_pr = np.mean([1 if e == 'right' else 0 for e in fast.escape_arm]) 
    slow_pr = np.mean([1 if e == 'right' else 0 for e in slow.escape_arm])

    ax2.bar([n-.15, n+.15],
        [fast_pr, slow_pr],
        color=[paper.maze_colors[maze], desaturate_color(paper.maze_colors[maze])],
        width=.3,
        edgecolor='k',
        )

    ax2.plot(
        [n-.30, n+.30],
        [all_pr, all_pr],
        lw=9, color='white',
        zorder=100,
    )

    ax2.plot(
        [n-.30, n+.30],
        [all_pr, all_pr],
        lw=5, color=[.5, .5, .5],
        zorder=100,
    )

clean_axes(f)
clean_axes(f2)

save_figure(f, os.path.join(paths.plots_dir, f"ToT by maze distribution"), svg=True)
save_figure(f2, os.path.join(paths.plots_dir, f"p(R) vs TOTO split 2"), svg=True)



# %%

"""
    Bin trials by ToT and look at p(R) in each pin
"""
f, axarr = plt.subplots(nrows=5, sharex=True, sharey=False, figsize=(14, 12))


bin_edges = [(a, b) for a,b in zip(bins[:-1], bins[1:])]
for n, (maze, trs) in enumerate(trials.datasets.items()):
    # Group trials
    binned = {be:[] for be in bin_edges}

    for i, t in trs.iterrows():
        for pre, post in bin_edges:
            if t.time_out_of_t >= pre and t.time_out_of_t < post:
                binned[(pre, post)].append(t)


    # Plot PR for group
    X, Y = [], []
    for (pre, post), ts in binned.items():
        if not ts: continue
        Y.append(np.mean([1 if t.escape_arm == 'right' else 0 for t in ts]))
        X.append(pre+(post-pre)/2)

        if len(ts) < 2:
            axarr[n].scatter(X[-1], Y[-1], color=[.4, .4, .4], alpha=.8, zorder=110, s=100)


    axarr[n].scatter(X, Y, color=paper.maze_colors[maze], s=200, lw=4, edgecolors='white', zorder=100)
    axarr[n].plot(X, Y, color=paper.maze_colors[maze], lw=2)

    axarr[n].set(ylabel=f'{maze}\np(R)')
axarr[-1].set(xlabel='ToT (s)')  # ylim=[-.1, 1.1]

f.tight_layout()
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f"ToT by maze p(R) binned by ToT"), svg=True)


# %%
"""
    Same as above but with equal number of observations per bin
"""
f, axarr = plt.subplots(nrows=5, sharex=True, sharey=False, figsize=(14, 12))

n_bins_maze = [16, 16, 16, 4, 9]

for n, (maze, trs) in enumerate(trials.datasets.items()):
    print(pd.qcut(trs.time_out_of_t, n_bins_maze[n]).value_counts().sort_index())
    edges = pd.qcut(trs.time_out_of_t, n_bins_maze[n]).value_counts().sort_index().index.values
    bin_edges = [(v.left, v.right) for v in edges]


    binned = {be:[] for be in bin_edges}
    for i, t in trs.iterrows():
        for pre, post in bin_edges:
            if t.time_out_of_t >= pre and t.time_out_of_t < post:
                binned[(pre, post)].append(t)

    X, Y = [], []
    for (pre, post), ts in binned.items():
        if not ts: continue
        Y.append(np.mean([1 if t.escape_arm == 'right' else 0 for t in ts]))
        X.append(pre+(post-pre)/2)

    axarr[n].scatter(X, Y, color=paper.maze_colors[maze], s=200, lw=4, edgecolors='white', zorder=100)
    axarr[n].plot(X, Y, color=paper.maze_colors[maze], lw=2)

    axarr[n].set(
        ylim = [np.min(Y) - .1, np.max(Y) + .1], ylabel=f'{maze}\np(R)'
    )

axarr[-1].set(xlabel='ToT (s)')  # ylim=[-.1, 1.1]

f.tight_layout()
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f"ToT by maze p(R) binned by ToT - equal N bins"), svg=True)

# %%
"""
    Same as above, but binning trials based on max escape speed

"""
f, axarr = plt.subplots(nrows=5, sharex=True, sharey=False, figsize=(14, 12))
n_bins_maze = [4, 4, 4, 3, 4]

KEY = 'stim_time_s'
YKEY = 'time_out_of_t'

for n, (maze, trs) in enumerate(trials.datasets.items()):
    # Get 'max' speed during escape
    trs['peak_speed'] = [np.percentile(s, 90) for s in trs.body_speed]

    edges = pd.qcut(trs[KEY].unique(), n_bins_maze[n]).value_counts().sort_index().index.values
    bin_edges = [(v.left, v.right) for v in edges]


    binned = {be:[] for be in bin_edges}
    for i, t in trs.iterrows():
        for pre, post in bin_edges:
            if t[KEY] >= pre and t[KEY] < post:
                binned[(pre, post)].append(t)

    X, Y = [], []
    for (pre, post), ts in binned.items():
        if not ts: continue
        if YKEY == 'accuracy':
            Y.append(np.mean([1 if t.escape_arm == 'right' else 0 for t in ts]))
        else:
            Y.append(np.mean([t[YKEY] for t in ts]))
        X.append(pre+(post-pre)/2)

    if n == 4 and YKEY == 'accuracy':
        Y = [1-y for y in Y]
    elif n == 3 and YKEY == 'accuracy': 
        #  in M4 accuracy is always 1
        axarr[n].axis('off')
        continue

    axarr[n].scatter(X, Y, color=paper.maze_colors[maze], s=200, lw=4, edgecolors='white', zorder=100)
    axarr[n].plot(X, Y, color=paper.maze_colors[maze], lw=2)

    axarr[n].set(
        ylim = [np.min(Y) - .1, np.max(Y) + .1], ylabel=f'{maze}\n\n{YKEY}'
    )

axarr[-1].set(xlabel=KEY)  # ylim=[-.1, 1.1]

f.suptitle(f'Trials split by {KEY}')
f.tight_layout()
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f"ToT by maze p(R) binned by {KEY} - equal N bins"), svg=True)
