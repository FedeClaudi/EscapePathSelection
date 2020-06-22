# %%
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import pandas as pd
import os
from math import sqrt
from scipy.signal import resample
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)


from fcutils.plotting.utils import create_figure, clean_axes, save_figure, set_figure_subplots_aspect
from fcutils.plotting.colors import *
from fcutils.plotting.plot_elements import plot_mean_and_error, rose_plot
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.plotting.colors import desaturate_color, makePalette
from fcutils.file_io.io import load_yaml
from fcutils.maths.geometry import calc_distance_from_point, calc_angles_with_arctan, calc_angle_between_vectors_of_points_2d
from fcutils.maths.filtering import line_smoother
from fcutils.maths.utils import derivative


from paper.dbase.TablesDefinitionsV4 import Session, Stimuli, TrackingData, Explorations
import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.utils.misc import resample_list_of_arrayes_to_avg_len


# %%
# Useful vars
save_fld = os.path.join(paths.plots_dir, 'shortcut')



conditions = ['Baseline', 'Condition 1', 'Condition2']
colors = [salmon, seagreen, goldenrod]

# Position of shelter
shelter = {'x':430, 'y':650, 'width':130, 'height':100, 'color':red, 'angle':0}
shelter_center = [
    shelter['x'] + shelter['width']/2,
    shelter['y'] + shelter['height']/2
]
# Position of 'block'
block = {'x':285, 'y':460, 'width':100, 'height':80, 'color':blue, 'angle':45}
block_center = [290, 525]

# Hand defined frame number  of when mouse is at block
at_block_frames = {
                '200210_CA8471_4': 65,
                '200210_CA8472_5': 85,
                '200210_CA8481_5': 18,
                '200210_CA8481_6': 70,
                '200210_CA8482_4': 150,
                '200210_CA8491_7': 82,
                '200210_CA8491_8': 24,
                '200225_CA8493_5': 96,
                '200227_CA8752_4': 90,
                '200227_CA8752_5': -1,
                '200227_CA8753_4': 60,
                '200227_CA8753_5': 82,
                '200227_CA8754_3': 115}


# Useful funcs
def add_rec_to_ax(ax, rec='shelter'):
    if rec == 'shelter':
        rec = shelter
    else:
        rec = block
    sh = Rectangle([rec['x'], rec['y']], rec['width'], rec['height'], angle=rec['angle'],
                        facecolor=rec['color'], alpha=.2, zorder=15 )
    ax.add_patch(sh)

def is_point_in_rec(x, y, rec='shelter'):
    if rec == 'shelter':
        rec = shelter
    else:
        rec = block
    if x < rec['x'] or x > rec['x']+rec['width']:
        return 0
    if y < rec['y'] or y > rec['y']+rec['height']:
        return 0
    return 1



# %%
# ---------------------------- Load notes and data --------------------------- #

notes_path = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\shortctu\\notes.yml'
notes = load_yaml(notes_path)

keep_sessions = [k[:-2] for k,n in notes.items() if n['overall']=='keep']
print('Keeping sessions: ', *keep_sessions)



print("Loading data")
N_frames = 20 * 30

trials = []
trials_by_sess = {}
# Get the tracking data after the stimulus for each trial
for session in keep_sessions:
    sname = f'{session}_1'
    stimuli = (Session * Stimuli * TrackingData * TrackingData.BodyPartData 
                            & f'session_name="{session}"' & "bpname='body'")
    neck = (Session *TrackingData * TrackingData.BodyPartData 
                            & f'session_name="{session}"' & "bpname='neck'").fetch('x', 'y')
    tail = (Session *TrackingData * TrackingData.BodyPartData 
                            & f'session_name="{session}"' & "bpname='tail_base'").fetch('x', 'y')


    note_trials = notes[sname]['trials']

    if len(stimuli) != len(note_trials):
        print(f'Wrong number of stimuli {session}.\nShould be {len(note_trials)} but it is {len(stimuli)}')
    else:
        print(f'All good {session}')

        # Get peri-stim tracking
        df = pd.DataFrame(stimuli.fetch())
        df['stim'] = df['overview_frame']
        data = dict(
            uid =[],
            session_name = [],
            mouse_id = [],
            stim = [],
            block_distance = [],
            x = [],
            y = [],
            neckx = [],
            necky = [],
            tailx = [],
            taily = [],
            direction_of_mvmt = [],
            orientation = [],
            speed = [],
            condition = [],
            stimulus_uid = [], 
            at_block = [],
            at_block_distance = [], # distance from animal's location when reaches block
            ori_rel_at_block = [], # orientation relative to orientation at block
        )
        for i, trial in df.iterrows():
            # Meteadata
            for key in ['uid', 'session_name', 'mouse_id', 'stim', 'stimulus_uid']:
                data[key].append(trial[key])
            
            # Tracking
            for key in ['x', 'y', 'speed', 'direction_of_mvmt']:
                data[key].append(trial[key][trial.stim : trial.stim + N_frames])

            data['neckx'].append(neck[0][0][trial.stim : trial.stim + N_frames])
            data['necky'].append(neck[1][0][trial.stim : trial.stim + N_frames])
            data['tailx'].append(tail[0][0][trial.stim : trial.stim + N_frames])
            data['taily'].append(tail[1][0][trial.stim : trial.stim + N_frames])


            data['orientation'].append(calc_angle_between_vectors_of_points_2d(
                                            data['x'][-1], data['y'][-1],
                                            data['neckx'][-1], data['necky'][-1],))


            # Stuff
            data['block_distance'].append(calc_distance_from_point(np.vstack([data['x'][-1], data['y'][-1]]), 
                                            block_center))

            data['condition'].append(notes[trial.session_name+'_1']['trials'][i][1])

            # Data relative to 'at_block
            if trial.stimulus_uid in at_block_frames.keys():
                at_block = at_block_frames[trial.stimulus_uid]

                data['at_block'].append(at_block)
                data['at_block_distance'].append(
                    calc_distance_from_point(
                        np.vstack([data['x'][-1], data['y'][-1]]), 

                        [data['x'][-1][at_block],
                            data['y'][-1][at_block]])
                )
                data['ori_rel_at_block'].append(
                            data['orientation'][-1] - data['orientation'][-1][at_block]
                )
            else:
                data['at_block'].append(None)
                data['at_block_distance'].append(None)
                data['ori_rel_at_block'].append(None)

        df = pd.DataFrame(data)
        trials.append(df)
        trials_by_sess[session] = df.copy()

trials = pd.concat(trials)

explorations = pd.DataFrame((Session * Explorations & "experiment_name='shortcut'"))





# %%

"""
    Plot each sessions' trials and all trials together
    by conditoin
"""

f, axarr = plt.subplots(ncols=5, nrows=2, figsize=(26, 15), sharex=True, sharey=True)
f2, axarr2 = plt.subplots(ncols=3, nrows=1, figsize=(18, 7), sharex=True, sharey=True)
f3, axarr3 = plt.subplots(ncols=3, nrows=1, figsize=(18, 7), sharex=True, sharey=True)
axarr = axarr.flatten()




got_to_shelt = {cond:[] for cond in conditions}
for ax, sess in zip(axarr, keep_sessions):
    # Plot exploration tracking
    exp = explorations.loc[explorations.session_name == sess].iloc[0]
    ax.plot(exp.body_tracking[:, 0], exp.body_tracking[:, 1], lw=.85, color=[.8, .8, .8], zorder=10)

    for i, trial in trials_by_sess[sess].iterrows():
        condition = notes[sess+'_1']['trials'][i][1]
        condition = notes[sess+'_1']['trials'][i][1]

        escape = is_point_in_rec(trial.x[-1], trial.y[-1])
        got_to_shelt[conditions[condition]].append(escape)

        # mark end point
        if condition == 2:
            lw = 4
            ax.scatter(trial.x[-1], trial.y[-1], color=red, zorder=99, s=100, ec='k')
            axarr2[condition].scatter(trial.x[-1], trial.y[-1], color=red, zorder=99, s=100, ec='k')
            axarr3[condition].scatter(trial.x[-1], trial.y[-1], color=red, zorder=99, s=100, ec='k')
        else:
            lw=2

        # Plot trials tracking
        if escape: 
            col = colors[condition]
        else:
            col = desaturate_color(colors[condition])

        ax.plot(trial.x, trial.y, color=colors[condition], lw=lw, zorder=90)
        axarr2[condition].plot(trial.x, trial.y, color=col, zorder=90)
        axarr3[condition].plot(trial.x, trial.y, color=col, zorder=90)

    ax.set(title=sess)
    ax.axis('off')
    add_rec_to_ax(ax)

axarr[-1].axis('off')

for ttl, ax in zip(conditions, axarr2):
    ax.set(title=ttl)
    ax.axis('off')
    add_rec_to_ax(ax)

for ttl, ax in zip(conditions, axarr3):
    ax.set(title=ttl, xlim=[230, 400], ylim=[420, 590])
    ax.axis('off')
    add_rec_to_ax(ax)

add_rec_to_ax(axarr2[-1], rec='block')
add_rec_to_ax(axarr3[-1], rec='block')

print(f'Condition 2 - {np.sum(got_to_shelt["Condition2"])}/{len(got_to_shelt["Condition2"])} to shelt')

clean_axes(f)
clean_axes(f2)
clean_axes(f3)

save_figure(f, os.path.join(save_fld, 'all_trials'))
save_figure(f2, os.path.join(save_fld, 'all_trials_by_cat'))
save_figure(f3, os.path.join(save_fld, 'all_trials_by_cat_only_block'))











# %%
"""
    More detailed look at behaviour on the condition 2 trials upon encountering the block
"""
c2trials = trials.loc[trials.condition == 2]


SHOW_TAIL = False
PLOT = False

f, axarr = plt.subplots(ncols=5, nrows=3, figsize=(25, 14), sharex=True, sharey=True)
f.suptitle('Tracking at block for condition 2 trials'
)
for ax in axarr.flat: ax.axis('off')

f2, axarr2 = plt.subplots(ncols=5, nrows=3, figsize=(25, 14),
                subplot_kw=dict(projection='polar'))
f2.suptitle('Orienting between at_bock and moved_away')

for ax in axarr2.flat: 
    ax.axis('off')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)



boxcolors = [indianred, orange, seagreen, seagreen, orange, 
        indianred, seagreen, orange, seagreen, white,
        indianred, seagreen, seagreen]



shelter_angles, at_block_angles, at_away_angles = [], [], []
# inset_axes = []
for ax, ax2, (i, trial), boxcolor in zip(axarr.flat, axarr2.flat, c2trials.iterrows(), boxcolors):
    # Get when the mouse arrives at the block
    at_block = int(trial.at_block)
    if at_block > 0:
        ax.scatter(trial.x[at_block], trial.y[at_block], ec='k', color=red, zorder=200)
        at_block_angles.append(trial.orientation[at_block])
  

    # Get when the mouse laves the circle at it's orientation
    radius = 30
    if at_block > 0:        
        moved_away = at_block + np.where(trial.at_block_distance[at_block:] >= radius)[0][0]
        at_away_angles.append(trial.orientation[moved_away])
        ax.scatter(trial.x[moved_away], trial.y[moved_away], ec='k', color=green, zorder=200)
    else:
        moved_away = -1

    sx, sy = shelter_center[0] - trial.x[at_block], shelter_center[1] - trial.y[at_block]
    shelter_angles.append(calc_angles_with_arctan(
                                    shelter_center[0]-trial.x[at_block],
                                    shelter_center[1]-trial.y[at_block]))

    # Plot colored lines    
    idxs = np.where(trial.block_distance < 150)[0]
    trialscolors = makePalette(powderblue, salmon, len(idxs))
    if PLOT:
        for idx, col in zip(idxs, trialscolors):
            if idx == at_block or idx == moved_away:
                lw, alpha, zorder, border = 6, 1, 99, True 


            elif idx > at_block and idx % 30 == 0:
                # continue
                lw, alpha, zorder, border = 4, 1, 90, True 
            else:
                lw, alpha, zorder, border = 4, .6, 80, False

            if border:
                # Plot dark border
                if SHOW_TAIL:
                    ax.plot([trial.tailx[idx], trial.x[idx]], [trial.taily[idx], trial.y[idx]], 
                                    color='k', lw=lw+1, alpha=1, zorder=85, solid_capstyle='round')
                ax.plot([trial.x[idx], trial.neckx[idx]], [trial.y[idx], trial.necky[idx]], 
                                color='k', lw=lw+1, alpha=1, zorder=85, solid_capstyle='round')

            # Plot colored line
            if SHOW_TAIL:
                ax.plot([trial.tailx[idx], trial.x[idx]], [trial.taily[idx], trial.y[idx]], 
                                color=col, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle='round')
            ax.plot([trial.x[idx], trial.neckx[idx]], [trial.y[idx], trial.necky[idx]], 
                            color=col, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle='round')


    # Add radius circle
    if at_block:
        circ = Circle([trial.x[at_block], trial.y[at_block]], radius=radius, 
                            color='k', fill=False, alpha=.4,zorder=16)
        ax.add_patch(circ)

    # Add ax border based on class
    ax.set(xlim=[220, 400], ylim=[410, 580], title=trial.stimulus_uid)
    # sh = Rectangle([220+5, 410+5], 400-220-10, 580-410-10, lw=4,
    #                     facecolor=white, ec=boxcolor, alpha=1, zorder=15 )
    # ax.add_patch(sh)

    # Fig2 add scatter
    if at_block > 0:
        th = np.radians(30)

        x = np.arange(len(trial.x[at_block:moved_away]))
        theta = np.radians(trial.orientation[at_block:moved_away])
        # ax2.plot(theta, x, color='k', lw=.75)
        c = makePalette(powderblue, salmon, len(x)+1)

        # plot lines
        ax2.plot([0, theta[0]], [0, x[-1]], lw=2, color='k', zorder=-1)
        ax2.plot([0, theta[0]-np.radians(180)], [0, x[-1]], lw=2, color='k', zorder=-1)
        
        ax2.plot([0, theta[0]+th], [0, x[-1]], lw=1, ls='--', color='k', zorder=-1)
        ax2.plot([0, theta[0]-th], [0, x[-1]], lw=1, ls='--', color='k', zorder=-1)

        ax2.plot([0, np.radians(shelter_angles[-1])], [0, x[-1]/2], lw=2, color=darkred, zorder=-1)

        # Scatter
        ax2.scatter(theta, x, color=[.4, .4, .4], zorder=9, s=150)
        ax2.scatter(theta, x, c=c, zorder=10, s=75)

        # Get first ang displacement above th
        dtheta = theta - theta[0]
        try:
            turn = np.where(np.abs(dtheta) >= th)[0][0]
            ax2.scatter(theta[turn], x[turn], color=darkred, lw=2, ec='k', s=165, zorder=20)
        except:
            pass
        
        # Style ax
        ax2.axis('on')
        ax2.set(xticks=np.radians([0, 90, 180, 270]), yticks=[])
        ax2.set_title(trial.stimulus_uid, pad=20)





# # Plot polar histogram of orientation relative to shelter at block
# pax = f.add_subplot(3, 5, 15, projection='polar')
# rose_plot(pax, np.radians(shelter_angles), as_hist=True, nbins=16, fill=False, edge_color=orange)
# pax.set_theta_direction(-1)
# pax.set(title='Angle to shelter at block')

# # Plot polar histogram of orientation at block
# pax = f.add_subplot(3, 5, 14, projection='polar')
# rose_plot(pax, np.radians(at_block_angles), as_hist=True, nbins=16, fill=False, edge_color=red, label='at block')
# rose_plot(pax, np.radians(at_away_angles), as_hist=True, nbins=16, fill=False, edge_color=green, label='at_awat')
# pax.set_theta_direction(-1)
# pax.set(title='Angles at block and at away')

set_figure_subplots_aspect(wspace=.5, hspace=.7, top=.9, bottom=.1)
save_figure(f, os.path.join(save_fld, 'con2_trials_detailed'))
save_figure(f2, os.path.join(save_fld, 'con2_trials_detailed_angles'))




# %%

