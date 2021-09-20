# %%
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.path import from_yaml
from fcutils.plot.figure import clean_axes

from paper import Explorations

from figures.first import M1, M2, M3, M4,M6
from figures._plot_utils import triple_plot
from figures.settings import dpi
from figures.bayes import Bayes
from figures.glm import GLM
from figures.second import fig_2_path

datasets = (M1, M2, M3, M4, M6)
# %%
def get_trips(rois, source, target, tracking):
    '''
        Given a list of roi indices for where the mouse is at for every frame, 
        it gets all the times that the mouse did a trip source -> target
    '''
    in_target = np.zeros(len(rois))
    in_target[rois == target] = 1

    at_target = np.where(np.diff(in_target)>0)[0]

    trips = []
    for tgt in at_target:
        # go back in time and get the last time the animal was at source
        try:
            at_source = np.where(rois[:tgt] == source)[0][-1]
        except  IndexError:
            # print('s1')
            continue

        # if the mouse gets to tgt any other time during the trip, ignore
        if np.any(rois[at_source:tgt-3] == target):
            # print('s2')
            continue

        if tgt - at_source < 50: 
            # print('s3')
            # too fast
            continue

        # get arm
        if np.min(tracking[at_source: tgt, 0]) <= 400:
            arm = 'left'
        else:
            arm = 'right'

        # check if there was an error
        if tracking[tgt, 0] < 200 or tracking[tgt, 0] > 800:
            # print('s4')
            continue

        # get distance travelled
        # x = tracking[at_source:tgt, 0]
        # y = tracking[at_source:tgt, 1]
        # dist = get_dist(x, y)
        dist = np.sum(tracking[at_source:tgt, 2]) * 0.22
        if dist < 25:
            # print('s5')
            continue  # too short

        trips.append((at_source, tgt, arm, dist))
    return trips

def plot_trips(tracking, rois, trips, ax=None):
    ax = ax or plt.subplots()[1]
    for trip in trips:
        trk = tracking[trip[0]:trip[1], :]
        rs = rois[trip[0]:trip[1]]

        ax.scatter(trk[0, 0], trk[0, 1], c='salmon')
        ax.scatter(trk[-1, 0], trk[-1, 1], c='green')

        if trip[2] == 'left':
            color = [.2, .2, .2]
        else:
            color = [.7, .7, .7]
        ax.plot(trk[::5, 0], trk[::5, 1], color=color, alpha=.25, zorder=-1)




# %%



# -------------------------------- exploration ------------------------------- #
# f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(16, 9), sharex=True, sharey=True)
# axarr=axarr.flatten()

counts_plot_kwargs = dict(
    shift=0.15, 
    zorder=100, 
    scatter_kws=None, 
    box_width=0.08,
    fill=0.1, 
    pad=0.0,
    spread=0.05,
    horizontal=False,
    show_kde=False
)


plot_kwargs = dict(
    shift=0.1, 
    zorder=100, 
    scatter_kws=None, 
    box_width=0.08,
    kde_normto=0.1,
    fill=0.1, 
    pad=0.0,
    spread=0.01,
    horizontal=False,
    
)

time_on_arm = dict(L=[], R=[])
all_summaries = {}
f, axes = plt.subplots(ncols=2, nrows=3, figsize=(16, 9), sharex=True)
for n, data in enumerate(datasets):
    # et ROI at each frame during explration
    store = dict(
        to=[],
        arm=[],
        start=[],
        end=[],
        duration=[],
        sess=[],
        distance = []
    )
    for sess in data.sessions:
        try:
            exploration = pd.Series((Explorations & f'session_name="{sess}"').fetch1())
        except Exception:
            print(f'No exploration for {sess}')
            continue

        # get time spent on each arm
        time_on_arm['L'].append(np.count_nonzero(exploration.body_tracking[:, 0] < 450))
        time_on_arm['R'].append(np.count_nonzero(exploration.body_tracking[:, 0] > 550))
    
        
        # to shelter
        trips_to_shelt = get_trips(exploration.maze_roi, 1, 0, exploration.body_tracking)
        for (start, end, arm, dist) in trips_to_shelt:
            store['to'].append('shelter')
            store['arm'].append(arm)
            store['start'].append(start)
            store['end'].append(end)
            store['duration'].append((end-start)/exploration.fps)
            store['sess'].append(sess)
            store['distance'].append(dist)

        
        # to threat
        trips_to_T = get_trips(exploration.maze_roi, 0, 1, exploration.body_tracking)
        for (start, end, arm, dist) in trips_to_T:
            store['to'].append('T')
            store['arm'].append(arm)
            store['start'].append(start)
            store['end'].append(end)
            store['duration'].append((end-start)/exploration.fps)
            store['sess'].append(sess)
            store['distance'].append(dist)

        # plot_trips(
        #     exploration.body_tracking, exploration.maze_roi,  trps, ax=axarr[n]
        # )

    summary = pd.DataFrame(store)
    all_summaries[data.name] = summary
    
    # plot data relative to shelter
    # counts
    L, R = [], []
    for sess in summary.sess.unique():
        counts = summary.loc[(summary.to == 'shelter')&(summary.sess==sess)].groupby('arm').count()
        noise = np.random.normal(0, .05, size=2)
        
        if 'left' not in counts.index:
            counts = [0, counts.sess[0]]
        elif 'right' not in counts.index:
            counts = [counts.sess[0], 0]
        else:
            counts = counts.sess
        # axes[0, 0].scatter([n-.15 + noise[0], n+.15+noise[1]], counts, c=[data.color, data.mice_color])
        L.append(counts[0])
        R.append(counts[1])

    triple_plot(
        n - .15, 
        L,
        axes[0, 0],
        invert_order = True,
        color=data.color,
        **counts_plot_kwargs
    )

    triple_plot(
        n + .15, 
        R,
        axes[0, 0],
        invert_order = False,
        color=data.mice_color,
        **counts_plot_kwargs
    )

    # duration
    triple_plot(
        n - .1, 
        summary.loc[(summary.to=='shelter')&(summary.arm=='left')].duration,
        axes[1, 0],
        invert_order = True,
        color=data.color,
        kde_kwargs=dict(bw=1),
        **plot_kwargs
    )

    triple_plot(
        n + .1, 
        summary.loc[(summary.to=='shelter')&(summary.arm=='right')].duration,
        axes[1, 0],
        invert_order = False,
        color=data.color,
        kde_kwargs=dict(bw=1),
        **plot_kwargs
    )

    # distance
    triple_plot(
        n - .1, 
        summary.loc[(summary.to=='shelter')&(summary.arm=='left')].distance,
        axes[2, 0],
        invert_order = True,
        color=data.color,
        kde_kwargs=dict(bw=8),

        **plot_kwargs
    )

    triple_plot(
        n + .1, 
        summary.loc[(summary.to=='shelter')&(summary.arm=='right')].distance,
        axes[2, 0],
        invert_order = False,
        color=data.color,
        kde_kwargs=dict(bw=8),

        **plot_kwargs
    )



    # plot data relative to threat
    # counts
    L, R = [], []
    for sess in summary.sess.unique():
        counts = summary.loc[(summary.to == 'T')&(summary.sess==sess)].groupby('arm').count()
        noise = np.random.normal(0, .05, size=2)
        
        if 'left' not in counts.index:
            counts = [0, counts.sess[0]]
        elif 'right' not in counts.index:
            counts = [counts.sess[0], 0]
        else:
            counts = counts.sess
        # axes[0, 0].scatter([n-.15 + noise[0], n+.15+noise[1]], counts, c=[data.color, data.mice_color])
        L.append(counts[0])
        R.append(counts[1])

    triple_plot(
        n - .15, 
        L,
        axes[0, 1],
        invert_order = True,
        color=data.color,
        **counts_plot_kwargs
    )

    triple_plot(
        n + .15, 
        R,
        axes[0, 1],
        invert_order = False,
        color=data.mice_color,
        **counts_plot_kwargs
    )
    # duration
    triple_plot(
        n - .1, 
        summary.loc[(summary.to=='T')&(summary.arm=='left')].duration,
        axes[1, 1],
        invert_order = True,
        color=data.color,
        kde_kwargs=dict(bw=1),
        **plot_kwargs
    )

    triple_plot(
        n + .1, 
        summary.loc[(summary.to=='T')&(summary.arm=='right')].duration,
        axes[1, 1],
        invert_order = False,
        color=data.color,
        kde_kwargs=dict(bw=1),
        **plot_kwargs
    )

    # distance
    triple_plot(
        n - .1, 
        summary.loc[(summary.to=='T')&(summary.arm=='left')].distance,
        axes[2, 1],
        invert_order = True,
        color=data.color,
        kde_kwargs=dict(bw=8),
        **plot_kwargs
    )

    triple_plot(
        n + .1, 
        summary.loc[(summary.to=='T')&(summary.arm=='right')].distance,
        axes[2, 1],
        invert_order = False,
        color=data.color,
        kde_kwargs=dict(bw=8),
        **plot_kwargs
    )



            
axes[0, 0].set(ylabel='# trips \n per mouse', title='Trips to SHELTER', xticks=np.arange(len(datasets)), ylim=[-2, 10])
axes[0, 1].set(title='Trips to THREAT', xticks=np.arange(len(datasets)), ylim=[-2, 10])
axes[1, 0].set(ylabel='duration (s) \n per trip', xticks=np.arange(len(datasets)), ylim=[0, 110])
axes[1, 1].set(xticks=np.arange(len(datasets)), ylim=[0, 110])
axes[2, 0].set(ylabel='distance (cm) \n per trip', xticks=np.arange(len(datasets)), xticklabels=[d.name for d in datasets], ylim=[0, 650])
axes[2, 1].set(xticks=np.arange(len(datasets)), xticklabels=[d.name for d in datasets], ylim=[0, 650])

clean_axes(f)

# %%
'''
    Plot histogram with total number of S and T trips per mouse across all mazes
'''
summary = pd.concat(all_summaries.values())
n_S_trips, n_T_trips = [], []

for sess in summary.sess.unique():
    n_S_trips.append(len(summary.loc[(summary.sess == sess)&(summary.to == 'shelter')]))
    n_T_trips.append(len(summary.loc[(summary.sess == sess)&(summary.to == 'T')]))

f, axes = plt.subplots(ncols=2, figsize=(16, 9))

# plot number of trips
for n, (counts, label, color) in enumerate(zip((n_S_trips, n_T_trips), ('shelter', 'threat'), ('salmon', [.6, .6, .6]))):
    axes[0].hist(counts, color=color, label=f'{label} trips', alpha=.5)

    boxes = axes[0].boxplot(
        counts, 
        positions=[-(n + 1)], 
        zorder=100, 
        widths=.5,
        showcaps=False,
        showfliers=False,
        patch_artist=True,
        boxprops = dict(color='k'),
        whiskerprops = dict(color=[.4, .4, .4], lw=2),
        medianprops = dict(color='k', lw=4),
        meanprops = dict(color='r', lw=2),
        manage_ticks=False,
        meanline=True,
        vert = False, 
    )

    for box in boxes["boxes"]:
        box.set(facecolor = color)

# plot time on each arm
for n, ((side, counts), color) in enumerate(zip(time_on_arm.items(), ('blue', 'green'))):
    counts = np.array(counts) / 40 / 60  # get counts in minues
    axes[1].hist(counts, color=color, label=f'{side} arm', alpha=.5)

    boxes = axes[1].boxplot(
        counts, 
        positions=[-(n + 1)*2], 
        zorder=100, 
        widths=1,
        showcaps=False,
        showfliers=False,
        patch_artist=True,
        boxprops = dict(color='k'),
        whiskerprops = dict(color=[.4, .4, .4], lw=2),
        medianprops = dict(color='k', lw=4),
        meanprops = dict(color='r', lw=2),
        manage_ticks=False,
        meanline=True,
        vert = False, 
    )

    for box in boxes["boxes"]:
        box.set(facecolor = color)

axes[0].set(ylabel='count', xlabel='# trips')
axes[0].legend()

axes[1].set(ylabel='count', xlabel='minutes')
axes[1].legend()

clean_axes(f)

axes[1].figure.savefig(fig_2_path / 'exploration_histograms.eps', format='eps', dpi=dpi)

    

# %%
mu = np.mean(n_S_trips + n_T_trips)
std = np.std(n_S_trips + n_T_trips)

print(f'Average number of trips: {mu:.2f} +/- {std:.2f}')

# %%
