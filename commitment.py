# %%
# Imports
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import pandas as pd
import os
from math import sqrt
from scipy.signal import resample
import random
from brainrender.colors import colorMap

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.colors import *
from fcutils.plotting.plot_elements import plot_mean_and_error
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.plotting.colors import desaturate_color


import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.utils.misc import resample_list_of_arrayes_to_avg_len

"""
    Set of plots to look at commitment to left vs right arm during escape
"""

# %%
# --------------------------------- Load data -------------------------------- #
print("Loading data")
params = dict(
    naive = None,
    lights = None, 
    tracking = 'threat',
    escapes_dur = True,
)

trials = TrialsLoader(**params)
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials

_mazes = get_mazes()

# %%
"""
    Plot trajectories on threat platform
"""
ONLY_CATWALK = True
ERRORBAR = True

f, axarr= plt.subplots(ncols=5, nrows=1, figsize=(22, 7))

for ax, (maze, trs) in zip(axarr, trials.datasets.items()):
    trs = trs.loc[trs.escape_arm != 'center']

    lcolor = paper.maze_colors[maze]
    rcolor = desaturate_color(lcolor, k=.2)

    left = {'x':[], 'y':[]}
    right = {'x':[], 'y':[]}

    for i, trial in trs.iterrows():
        if ONLY_CATWALK and trial.body_xy[0, 1] > 230: continue 

        color = lcolor if trial.escape_arm == 'left' else rcolor
        if not ERRORBAR:
            alpha = .4
        else:
            alpha = .1

        ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1], 
                        color=desaturate_color(color), lw=alpha)


        if trial.escape_arm == 'left':
            left['x'].append(trial.body_xy[:, 0])
            left['y'].append(trial.body_xy[:, 1])
        else:
            right['x'].append(trial.body_xy[:, 0])
            right['y'].append(trial.body_xy[:, 1])

    # Get the average trajectory
    for samples, color in zip([left, right], [lcolor, rcolor]):
        mean_dur = np.mean([len(x) for x in samples['x']]).astype(np.int32)
        pad = 100
        x = resample_list_of_arrayes_to_avg_len(samples['x'], interpolate=True)
        y = resample_list_of_arrayes_to_avg_len(samples['y'], interpolate=True)

        meanx = np.nanmean(x, 0)[3:-3]
        stdx = np.nanstd(x, 0)[3:-3]
        meany = np.nanmean(y, 0)[3:-3]
        stdy = np.nanstd(y, 0)[3:-3]

        if ERRORBAR:
            plot_mean_and_error(meany, stdx, ax, x=meanx, color=color,
                                        err_alpha=.5)

        ax.plot(meanx, meany, lw=9, color=white, zorder=101)
        ax.plot(meanx, meany, lw=7, color=color, zorder=101)


        # ax.fil_between(meanx-stdx, )


    # Clean up and save
    ax.axis('off')
    ax.set(title=maze)

f.suptitle('Mean trajectory on threat platform')
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'mean trajectory on threat'))



# %%
"""
    Tracking example trials showing the animal's body axis at each frame
"""
f, axarr= plt.subplots(ncols=5, nrows=1, figsize=(22, 7))

for ax, (maze, trs) in zip(axarr, trials.datasets.items()):
    trs = trs.loc[trs.escape_arm != 'center']

    lcolor = paper.maze_colors[maze]
    rcolor = desaturate_color(lcolor, k=.2)

    
    examples = [
        trs.loc[trs.escape_arm == 'left'].iloc[-1],
        trs.loc[trs.escape_arm == 'right'].iloc[-1]
    ]

    for trial, color in zip(examples, [lcolor, rcolor]):
        nx, ny = trial.neck_xy[:, 0], trial.neck_xy[:, 1]
        bx, by = trial.body_xy[:, 0], trial.body_xy[:, 1]
        tx, ty = trial.tail_xy[:, 0], trial.tail_xy[:, 1]

        # Plot every N thick white
        ax.plot([nx[1::8], bx[1::8]], [ny[1::8], by[1::8]], color=white, lw=4, zorder=1,
                        solid_capstyle='round')
        ax.plot([bx[1::8], tx[1::8]], [by[1::8], ty[1::8]], color=white, lw=4, zorder=1,
                        solid_capstyle='round')

        # Plot every N
        # ax.scatter(nx[1::8], ny[1::8], color=red, lw=3, zorder=100, marker="d")
        ax.plot([nx[1::8], bx[1::8]], [ny[1::8], by[1::8]], color=color, lw=3, zorder=3,
                        solid_capstyle='round')
        ax.plot([bx[1::8], tx[1::8]], [by[1::8], ty[1::8]], color=color, lw=3, zorder=2,
                        solid_capstyle='round')

        # Plot all trials
        ax.plot([nx, bx], [ny, by], color=color, lw=.8, alpha=.7, zorder=0,
                        solid_capstyle='round')
        ax.plot([bx, tx], [by, ty], color=color, lw=.8, alpha=.7, zorder=0,
                        solid_capstyle='round')


        

    # Clean up and save
    ax.axis('off')
    ax.set(title=maze)

f.suptitle('Example trajectory on threat platform')
clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, 'example trajectory on threat'))







# %%
"""
    Plot heading relative to shelter

"""
ONLY_CATWALK = True


f, axarr= plt.subplots(nrows=5,figsize=(18, 12), sharex=True, sharey=True)

for ax, (maze, trs) in zip(axarr, trials.datasets.items()):
    trs = trs.loc[trs.escape_arm != 'center']

    lcolor = paper.maze_colors[maze]
    rcolor = desaturate_color(lcolor, k=.2)

    lthetas, rthetas = [], []
    for i, trial in trs.iterrows():
        if ONLY_CATWALK and trial.body_xy[0, 1] > 230: continue 

        # bdy dir of mvmt = 90 means going towards the shelter
        theta = trial.body_dir_mvmt[1:] 

        theta = pd.Series(theta).interpolate()

        
        if trial.escape_arm == 'left':
            lthetas.append(theta)
        else:
            rthetas.append(theta)

    mean_dur = 80
    lt = np.vstack([resample(pd.Series(X).interpolate(), mean_dur) for X in lthetas])
    rt = np.vstack([resample(pd.Series(X).interpolate(), mean_dur) for X in rthetas])

    mean_theta_l = np.zeros(mean_dur)
    mean_theta_r = np.zeros_like(mean_theta_l)
    std_theta_l = np.zeros(mean_dur)
    std_theta_r = np.zeros_like(std_theta_l)

    for i in np.arange(mean_dur):
        mean_theta_l[i] = np.nanmean(lt[:, i])
        mean_theta_r[i] = np.nanmean(rt[:, i])

        std_theta_l[i] = np.nanstd(lt[:, i])
        std_theta_r[i] = np.nanstd(rt[:, i])

    plot_mean_and_error(mean_theta_l, std_theta_l, ax, color=lcolor)
    plot_mean_and_error(mean_theta_r, std_theta_r, ax, color=rcolor)
    # ax.plot(mean_theta_l, color=lcolor)
    # ax.plot(mean_theta_r, color=rcolor)

    ax.axhline(90, lw=2, color=[.4, .4, .4], zorder=-1)
    ax.axhline(90 + _mazes[maze]['left_path_angle'], lw=2, color=[.6, .6, .6], ls='--', zorder=-1)
    ax.axhline(90 - _mazes[maze]['right_path_angle']+90, lw=2, color=[.6, .6, .6], ls='--', zorder=-1)

    # ax.set(ylim=[-_mazes[maze]['right_path_angle']-25, _mazes[maze]['left_path_angle']+25])

# %%
"""
    Plot binned heading direction
"""
QUIVER = True
ARM = 'right'
ONLY_CATWALK = True

X = np.arange(40, 65)
Y = np.arange(10, 36)


f, axarr= plt.subplots(ncols=5, figsize=(22, 9), sharex=True, sharey=True)

for ax, (maze, trs) in zip(axarr, trials.datasets.items()):
    bins = {(x, y):[] for x in X for y in Y}

    trs = trs.loc[trs.escape_arm != 'center']

    lcolor = paper.maze_colors[maze]
    rcolor = desaturate_color(lcolor, k=.2)

    left = {'x':[], 'y':[]}
    right = {'x':[], 'y':[]}

    lthetas, rthetas = [], []
    for i, trial in trs.iterrows():
        if ONLY_CATWALK and trial.body_xy[0, 1] > 230: continue 

        x = (trial.body_xy[:, 0] / 10).astype(np.int32)
        y = (trial.body_xy[:, 1] / 10).astype(np.int32)

        if ARM is not None:
            if trial.escape_arm != ARM: continue

        for x, y, t in zip(x, y, trial.body_dir_mvmt-90):
            if np.isnan(t): continue
            if (x, y) not in bins.keys(): continue
            bins[(x, y)].append(t)
        # ax.plot(x, y)


    x = [x for (x, y), t in bins.items() if len(t)>3]
    y = [y for (x, y), t in bins.items() if len(t)>3]
    t = np.array([np.mean(t) for t in bins.values() if len(t)>3])
    s = [len(t) for t in bins.values() if len(t) > 3]

    if not QUIVER:
        ax.scatter(x, y, s=80, c=t, cmap='bwr', vmin=-_mazes[maze]['right_path_angle'], vmax=_mazes[maze]['left_path_angle'],
                        ec='k', lw=.5, zorder=-1)
    else:
        ax.quiver(x, y, np.cos(np.radians(t+90)), np.sin(np.radians(t+90)), t, width=.01, scale=15, angles ='xy', 
                cmap='bwr', ec=[.2, .2, .2], lw=.25)

    ax.set(title=maze)
    ax.axis('off')

f.suptitle('Average binned direction of movement on threat platform')
save_figure(f, os.path.join(paths.plots_dir, f'mean angle on threat {"quiver" if QUIVER else ""} - {ARM}'))

# %%
"""
    Bin each trajectory in time, and plot the average trajectory
"""
ONLY_CATWALK = True
f, axarr= plt.subplots(ncols=5, nrows=2, figsize=(22, 7))
f.suptitle('Average haeding directoin')

mazes_datas = {}

for m, (maze, trs) in enumerate(trials.datasets.items()):
    trs = trs.loc[trs.escape_arm != 'center']

    lcolor = paper.maze_colors[maze]
    rcolor = desaturate_color(lcolor, k=.2)

    data = {'left':{'x':[], 'y':[], 't':[]}, 'right':{'x':[], 'y':[], 't':[]}}
    for i, trial in trs.iterrows():

        if ONLY_CATWALK and trial.body_xy[0, 1] > 230: continue 

        if trial.escape_arm == 'left':
            ax = axarr[0, m]
        else: 
            ax = axarr[1, m]

        ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1], lw=2, color=[.9, .9, .9], zorder=-10)

        # bdy dir of mvmt = 90 means going towards the shelter
        theta = trial.body_dir_mvmt
        theta = pd.Series(theta).interpolate()

        data[trial.escape_arm]['x'].append(trial.body_xy[:, 0])
        data[trial.escape_arm]['y'].append(trial.body_xy[:, 1])
        data[trial.escape_arm]['t'].append(theta)

    mazes_datas[maze] = data
    
    
    # Plot the average heading direction per bin
    # make sure to use the same number of trials
    ntrials = np.min([len(data['left']['x']), len(data['right']['x'])])

    for color, (arm, xyt), ax in zip([lcolor, rcolor], data.items(), [axarr[0, m], axarr[1, m]]):
        # Select subset of trials
        idx = random.choices(np.arange(len(xyt['x'])), k=ntrials)
        
        for key in xyt.keys():
            xyt[key] = [xyt[key][i] for i in idx]

        if len(xyt['x']) != ntrials: raise ValueError

        x = np.nanmean(resample_list_of_arrayes_to_avg_len(xyt['x']), 0)
        y = np.nanmean(resample_list_of_arrayes_to_avg_len(xyt['y']), 0)
        t = np.nanmean(resample_list_of_arrayes_to_avg_len(xyt['t']), 0)

        l = len(x)
        bins = np.arange(20, l, 11)

        for n, bin in enumerate(bins[:-1]):
            meanx = np.nanmean(x[bin:bins[n+1]])
            meany = np.nanmean(y[bin:bins[n+1]])
            meant = np.nanmean(t[bin:bins[n+1]])

            # Plot arrow with meadian heading direction
            dx = np.cos(np.radians(meant))
            dy = np.sin(np.radians(meant))

            ax.arrow(meanx, meany, dx*12, dy*12, width=3, head_width=6, ec='k', lw=.2, fc=color)

            # Plot circle segment with low and high quartile range of theta
            lowt = np.percentile(t[bin:bins[n+1]], 1)
            hight = np.percentile(t[bin:bins[n+1]], 99)

            arc = Wedge((meanx, meany),  30, width=15, theta1=lowt, theta2=hight, fc=color, alpha=.4)
            ax.add_patch(arc)

            ax.axis('off')
            ax.set(title=maze if arm == 'left' else None, xlim=[460, 540], ylim=[150, 330])


save_figure(f, os.path.join(paths.plots_dir, f'mean heading dir on T binned'))

# %%
"""
    Predict escape arm as a function of heading direction

    Using mazes_datas generated above
"""
import random 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


N_repeats = 1000
bins = None


f, axarr = plt.subplots(ncols=6, sharey=True,figsize=(22, 7))
f.suptitle(f'Logistic regression: predict arm from dir of mvmt\n {N_repeats} repeats per maze')

for ax, (maze, data) in tqdm(zip(axarr[1:], mazes_datas.items())):

    # --------------------------------- prep data -------------------------------- #
    x = data['left']['x'] + data['right']['x']
    y = data['left']['y'] + data['right']['y']
    t = data['left']['t'] + data['right']['t']
    arm = [0 for l in np.arange(len(data['left']['x']))] + \
                [1 for r in np.arange(len(data['right']['x']))]

    for xx, yy in zip(x,y):
        axarr[0].plot(xx, yy, lw=.25, color=[.7, .7, .7])

    # Get single frames
    X, Y, A, T = [], [], [], []
    for xx, yy, tt, a in zip(x, y, t, arm):
        X.extend(list(xx))
        Y.extend(list(yy))
        T.extend(list(tt))
        A.extend([a for i in xx])

    # Keep the same number of frames for left and right arm
    df = pd.DataFrame(dict(x=X, y=Y, arm=A, t=T)).dropna()

    left = df.loc[df.arm == 0]
    right = df.loc[df.arm == 1]
    
    if len(left) < len(right):
        right = right.sample(len(left))
    else:
        left = left.sample(len(right))
    df = pd.concat([left, right])
    df = df.loc[(df.y > 150)&(df.y < 330)]
    
    # --------------------------------- fit model -------------------------------- #
    # Fit the model 10 times and take average
    allcorrect, allincorrect = [], []
    for repeat in np.arange(N_repeats):
        # Create training and test sets
        xtrain, xtest, ytrain, ytest = train_test_split(df[['x', 'y', 't']], df[['arm']], test_size=0.33)
        
        # Fit model ad predict test
        model = LogisticRegression().fit(xtrain[['t']], ytrain)
        ypred = model.predict(xtest[['t']])

        # Get which predictions are correct
        correct = [1 if yp==y else 0 for yp, y in zip(ypred, ytest['arm'].values)]

        # Bin model's accuracy versus Y coord
        if bins is None:
            counts, bins = np.histogram(xtest.y)
        else:
            counts, _ = np.histogram(xtest.y)

        counts = counts.astype(np.float64)
        counts[counts < 10] = np.nan #  make sure only bins with enough samples are kept

        # Get proportion of correct vs incorrect
        correct_counts, _ = np.histogram(xtest.y.iloc[np.where(np.array(correct))], bins=bins)
        incorrect_counts, _ = np.histogram(xtest.y.iloc[np.where(1-np.array(correct))], bins=bins)

        allcorrect.append(correct_counts/counts)
        allincorrect.append(incorrect_counts/counts)

    # Plot
    x = bins[0:-1] # + np.cumsum(np.diff(bins)/2)
    

    colors = [colorMap(p, 'Greens', vmin=0, vmax=1) for p in np.mean(allcorrect, 0)]

    ax.barh(x, np.mean(allcorrect, 0), xerr=np.std(allcorrect, 0), 
                color=colors, height=12, left=0, alpha=.75, 
                orientation='horizontal', align='edge')

    ax.axvline(0.5, lw=2, color=[.6, .6, .6], ls='--', zorder=-1)
    ax.axvline(0.25, lw=1, color=[.6, .6, .6], ls=':', zorder=-1)
    ax.axvline(0.75, lw=1, color=[.6, .6, .6], ls=':', zorder=-1)

    ax.set(title=maze, xlabel='fraction correct', xticks=[0, .5, 1], xlim=[0, 1])

# axarr[0].axhline(150)
# axarr[0].axhline(330)

axarr[0].axis('off')
axarr[0].set(ylabel='Y coordinate')

clean_axes(f)
save_figure(f, os.path.join(paths.plots_dir, f'predict arm from theta'))





# %%
