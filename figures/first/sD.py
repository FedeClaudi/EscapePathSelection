# %%
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.path import from_yaml

from paper import Explorations, Stimuli

from figures.first import M1, M2, M3, M4,M6,  fig_1_path
from figures._plot_utils import generate_figure, triple_plot
from figures.settings import dpi
from figures.bayes import Bayes
from figures.glm import GLM

datasets = (M1, M2, M3, M4, M6)


'''
    Plot exploration bias for mazes M1 to M4

    Plot effect of arm of origin on p(R)

'''


# %%
maze_components_lookup = from_yaml(r'C:\Users\Federico\Documents\GitHub\EscapePathSelection\paper\dbase\rois_lookup.yml')
maze_components_lookup = {v:k for k,v in maze_components_lookup.items()}


maze_components_locations = from_yaml(r'C:\Users\Federico\Documents\GitHub\EscapePathSelection\paper\dbase\MazeModelROILocation.yml')
left_components = [k for k, (x,y) in maze_components_locations.items() if x < 480 ]
right_components = [k for k, (x,y) in maze_components_locations.items() if x > 520 ]

# %%
plot_kwargs = dict(
    shift=0.04, 
    zorder=100, 
    scatter_kws=None, 
    kde_kwargs=dict(bw=0.01),
    box_width=0.04,
    kde_normto=0.1,
    fill=0.1, 
    pad=0.0,
    spread=0.025,
    horizontal=False
)

ax = generate_figure()
# -------------------------------- exploration ------------------------------- #
for n, data in enumerate(datasets):

    # et ROI at each frame during explration
    L, R = [], []  # time spent in L/R arms
    for sess in data.sessions:
        try:
            exploration = pd.Series((Explorations & f'session_name="{sess}"').fetch1())
        except Exception:
            print(f'No exploration for {sess}')
            continue


        l = np.count_nonzero(exploration.body_tracking[:, 0] < 450)
        r = np.count_nonzero(exploration.body_tracking[:, 0] > 550)

        # normalize by path length
        l = l / data.maze['left_path_length'] * data.maze['right_path_length']

        L.append(l/(l+r))
        R.append(r/(l+r))

    # plot
    triple_plot(
        n - .2, 
        L,
        ax,
        invert_order = True,
        color=data.color,
        **plot_kwargs
    )

    triple_plot(
        n + .2, 
        R,
        ax,
        color=data.color,
        **plot_kwargs
    )
    
ax.axhline(.5, lw=2, ls=':', color=[.6, .6, .6], zorder=-1)
_ = ax.set(ylim=[0, 1], xticks=np.arange(len(datasets)), xticklabels=[m.name for m in datasets],
    xlabel='maze', ylabel=r'arm occupancy $\frac{l}{l+r}$ norm. by length')
ax.figure.savefig(fig_1_path / 'panel_S_D_explorations.eps', format='eps', dpi=dpi)

# %%
# ------------------------------- arm of orign ------------------------------- #
bayes = Bayes()
ax = generate_figure(figsize=(16,9))

plot_kwargs = dict(
    shift=0.005, 
    zorder=100, 
    scatter_kws=None, 
    kde_kwargs=dict(bw=0.025),
    box_width=0.002,
    kde_normto=0.005,
    fill=0.1, 
    pad=0.0,
    spread=0.0005,
    horizontal=False
)

X = []
_data = dict(session=[], pR=[], side=[])
_skipped = []
for n, data in enumerate(datasets):
    n = n / 20
    X.append(n)
    L_origin = data.trials.loc[data.trials.origin_arm == 'left']
    R_origin = data.trials.loc[data.trials.origin_arm == 'right']
    colors = (data.color, data.mice_color)
    labels = ('left origin', 'right origin')
    sides = ['left', 'right']

    means = []
    for side, trials, color, label in zip(sides, (L_origin, R_origin), colors, labels):
        pR = []
        for sess in data.sessions:
            sess_trials = trials.loc[trials.session_name ==  sess]
            if sess_trials.empty:
                print(f'Skipping {sess} - {side.upper()}')
                _skipped.append(sess)
                continue
            nR = len(sess_trials.loc[sess_trials.escape_arm == 'right'])
            pR.append(nR / len(sess_trials))
            
            # append to data for stats tests
            _data['session'].append(sess)
            _data['pR'].append(pR[-1])
            _data['side'].append(side)

        shift = .005 if side == 'right' else-.005
        triple_plot(
            n + shift,
            pR,
            ax,
            invert_order = side == 'left',
            color=color,
            **plot_kwargs
        )
        ax.scatter(
            n + shift,
            np.mean(pR),
            s=200,
            color='w',
            lw=5,
            edgecolors=color,
            zorder=200
        )
        means.append((n + shift,
            np.mean(pR)))

    ax.plot(
        [means[0][0], means[1][0]], 
        [means[0][1], means[1][1]],
        lw=6, color=[.4, .4, .4], zorder=-1
    )


_ = ax.set(ylim=[-0.02, 1.02], xticks=X, 
            xticklabels=[m.name for m in datasets], ylabel='p(R)')
ax.figure.savefig(fig_1_path / 'panel_S_D_ArmOfOrigin.eps', format='eps', dpi=dpi)

# %%
# do a repeated measures anova to check for effect of arm
_data = pd.DataFrame(_data)

for sess in set(_skipped):
    _data = _data.loc[_data.session != sess]

from statsmodels.stats.anova import AnovaRM

#perform the repeated measures ANOVA
print(AnovaRM(data=_data, depvar='pR', subject='session', within=['side']).fit())



# %%
# # fit GLM on all data

# for data in datasets:
#     angle_ratio = data.maze['left_path_angle']/ data.maze['right_path_angle']
#     data.trials['angle_ratio'] = angle_ratio
#     data.trials['geodesic_ratio'] = data.maze['ratio']
#     data.trials['outcomes'] = [1 if arm == 'right' else 0 for arm in data.trials.escape_arm.values]
#     data.trials['origin'] = [1 if arm == 'right' else 0 for arm in data.trials.origin_arm.values]
#     data.trials['maze'] = data.maze['maze_name'].upper()

# # glm_data = pd.DataFrame(glm_data)
# glm_data = pd.concat([d.trials for d in datasets]).reset_index()
# glm_data = glm_data[['maze', 'angle_ratio', 'origin', 'geodesic_ratio', 'outcomes']]

# glm1 = GLM(glm_data, ['angle_ratio', 'geodesic_ratio', 'origin'], ['pR'])
# R2_one, _ = glm1.fit_bootstrapped(repetitions=10000,)

# glm2 = GLM.from_datasets(datasets)
# R2_two, _ = glm2.fit_bootstrapped(repetitions=10000)


# ax = generate_figure()
# ax.hist(R2_one, bins=np.linspace(.9985, .9999, 30), label='$r^2$ - with origin',  color=[.4, .4, .4])
# ax.hist(R2_two, bins=np.linspace(.9985, .9999, 30), label='$r^2$ - without origin', color='salmon', alpha=.6)
# ax.legend()


# %%
'''
    Average exploration duration and number of stimuli per mouse
'''
datasets = (M1, M2, M3, M4, M6)
expl_durations, n_stimuli = [], []
for data in datasets:
    for sess in data.uid.unique():
        sess_stimuli = pd.DataFrame((Stimuli & f'uid={sess}' & 'overview_frame>0'))
        n_stimuli.append(len(sess_stimuli))

        exploration = pd.Series((Explorations & f'uid="{sess}"').fetch1())
        expl_durations.append(exploration.duration_s/60)


axes = generate_figure(ncols=2, figsize=(16, 8))

axes[0].hist(expl_durations)
axes[1].hist(n_stimuli)
axes[0].set(title='Exploration duration')
axes[1].set(title='Numbe of stimuli')


print(f'Exploration duration average: {np.mean(expl_durations):.2f} - sdev {np.std(expl_durations):.2f}')

print(f'Number of tirals average: {np.mean(n_stimuli):.2f} - sdev {np.std(n_stimuli):.2f}')
# %%
