# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
import paper
from paper import paths
import os
from rich.progress import track

from paper.dbase.TablesDefinitionsV4 import Explorations, Session
from paper.dbase.utils import convert_roi_id_to_tag, plot_rois_positions
from paper.paths import plots_dir
from paper import maze_colors
from paper.helpers.mazes_stats import get_mazes
from paper.utils.explorations import get_maze_explorations

from fcutils.plotting.utils import clean_axes, save_figure
from fcutils.plotting.plot_distributions import plot_kde
from fcutils.plotting.colors import *
from fcutils.file_io.utils import check_create_folder

from behaviour.utilities.signals import get_times_signal_high_and_low


# %%

def get_session_time_on_arms(exploration, normalize = False, maze=None):
    """

        Given one entry of Explorations() this returns the time spent on each
        of the arms during the exploration. If normalise=True, the time per arm
        is normalized by the relative length of each arm (relative to the right path)
    """
    tracking = exploration.body_tracking

    # Get time spent on each path in seconds normalized by total duration of explorations
    on_left = len(tracking[tracking[:, 0]  < 450])  / exploration.fps
    on_right = len(tracking[tracking[:, 0]  > 550])  / exploration.fps
    tot = on_left + on_right
    if not tot: return None
    on_left /= tot
    on_right /= tot


    # NOrmalize
    if normalize: 
        metadata = get_mazes()[f'maze{maze}']
        ratio = metadata['right_path_length'] / metadata['left_path_length']
        on_left *= ratio

    return on_left, on_right



# %%
"""
    Plot the time spent on each path for each exploration
"""
mazes = [1, 2, 3, 4, 6]
NORMALIZE = True


f, axarr = plt.subplots(ncols = len(mazes), figsize=(30, 8))

for maze, ax in zip(mazes, axarr):
    explorations = get_maze_explorations(maze,  naive=None, lights=1)

    occupancy = []

    for i, exp in tqdm(explorations.iterrows()):
        occ = get_session_time_on_arms(exp, normalize=NORMALIZE, maze=maze)
        if occ is None: continue
        occupancy.append(occ)

    occupancy = pd.DataFrame(dict(
        l = [l for l, r in occupancy],
        r = [r for l, r in occupancy]
    ))



    ax.plot(occupancy.T, 'o-',  ms=10, color=[.7, .7, .7])
    plot_kde(data=occupancy.l, vertical=True, normto=-.2, ax=ax, color=maze_colors[f'maze{maze}'], alpha=.1)
    plot_kde(data=occupancy.r, z=1, vertical=True, normto=.2, ax=ax, color=maze_colors[f'maze{maze}'], alpha=.1)
    ax.plot([0, 1], occupancy.median(), '-o', ms=25, color=maze_colors[f'maze{maze}'], lw=3)

    ax.set(title=f'Maze {maze}')
clean_axes(f)

save_figure(f, os.path.join(plots_dir, f"exploration_arm_occupancy{'_norm' if NORMALIZE else ''}"))

# %%

"""
    Plot the fraction fo IN/OUT trips between the left and right path for each maze
"""

f, ax = plt.subplots()

only = 'in'

for n, maze in enumerate(mazes):
    Y = [0]
    explorations = get_maze_explorations(maze,  naive=None, lights=1)
    for i, e in explorations.iterrows():

        # Get left and right outwards and inwards trips
        tracking = e.body_tracking

        left = np.zeros_like(tracking[:, 0])
        left[tracking[:, 0] < 450] = 1
        l_starts, l_ends = get_times_signal_high_and_low(left, th=.5)

        right = np.zeros_like(tracking[:, 0])
        right[tracking[:, 0] > 550] = 1
        r_starts, r_ends = get_times_signal_high_and_low(right, th=.5)

        n_trips = dict(
            left = dict(outward = 0, inward = 0),
            right = dict(outward = 0, inward = 0),

        )
        
        for start, end in zip(l_starts, l_ends):
            if end - start < 100: continue
            if tracking[start, 1]  > 500:
                n_trips['left']['outward'] += 1
            else:
                n_trips['left']['inward'] += 1

        
        for start, end in zip(r_starts, r_ends):
            if end - start < 100: continue
            if tracking[start, 1]  > 500:
                n_trips['right']['outward'] += 1
            else:
                n_trips['right']['inward'] += 1

        # Summarise
        if only is None: # consider all trips
            L, R = n_trips['left']['outward']+n_trips['left']['inward'], n_trips['right']['outward']+n_trips['right']['inward']
        elif only == 'out':
            L, R = n_trips['left']['outward'], n_trips['right']['outward']
        elif only == 'in':
            L, R = n_trips['left']['inward'], n_trips['right']['inward']
        else:
            raise ValueError

        if not L and not R:
            continue
        elif not L:
            y = 1
        elif not R:
            y = 0
        else:
            y = R/(L+R)
        
        # Plot scatter
        ax.scatter(np.random.normal(n, 0.001), y, s=40, color=[.4, .4, .4], alpha=.5)
        Y.append(y)

    # Plot KDE and mean
    plot_kde(data=Y, vertical=True,  z=n, normto=.3, ax=ax, color=maze_colors[f'maze{maze}'], alpha=.1, kde_kwargs=dict(bw=.02))
    ax.scatter(n, np.mean(Y), s=160, color='red', zorder=99)


ax.axhline(0.5, lw=2, ls='--', color='k')

ax.set(ylim=[-.1, 1.1], title=f'Fraction of {only} trips per maze [exploration]', ylabel='fraction of trips', 
        xticklabels=[f'maze{m}' for m in mazes], xticks=np.arange(len(mazes)))

clean_axes(f)



save_figure(f, os.path.join(plots_dir, f"exploration_trips_fraction{only}"))

# %%
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# %%

"""
    Plot exploration tracking as a function of time for each mouse

"""

individual = False

outfld = os.path.join(plots_dir, 'exploration')
check_create_folder(outfld)

for n, maze in enumerate(mazes):
    Y = [0]
    explorations = get_maze_explorations(maze,  naive=None, lights=1)
    N = len(explorations)

    maze_fld = os.path.join(outfld, f'maze {maze}')
    check_create_folder(maze_fld)

    if not individual:
        fig = plt.figure(figsize=(24*2, 14*2))
    for i,e in explorations.iterrows():
        if not individual:
            ax = fig.add_subplot(5 , 9, i+1, projection='3d')
        else:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')

        # Get tracking with only time on arms
        tracking = e.body_tracking
        in_shelt = np.where(e.maze_roi == 0)[0]
        in_threat = np.where(e.maze_roi == 1)[0]

        trk = tracking.copy()
        # trk[in_shelt] = np.nan
        # trk[in_threat] = np.nan

        time = np.arange(len(trk))
        ones = np.ones(len(trk))*-40

        # Get in and out of shelt times
        in_shelt = np.zeros(len(time))
        in_shelt[e.maze_roi == 0] = 1
        in_shelt[:3000] = 0
        in_outs = get_times_signal_high_and_low(in_shelt, th=.5)
        if len(in_outs[0]):
            first_shelt = in_outs[0][0]
        else:
            first_shelt = len(tracking)-1

        # Plot tracking
        ax.plot(trk[:first_shelt, 0], trk[:first_shelt, 1], time[:first_shelt], lw=3, color='k')
        ax.plot(trk[first_shelt:, 0], trk[first_shelt:, 1], time[first_shelt:], lw=3, color='g')
        ax.scatter(trk[first_shelt, 0], trk[first_shelt, 1], time[first_shelt], lw=3, color='k', s=200)


        # Plot shelter enter exits
        # ax.scatter(tracking[in_outs[0], 0], tracking[in_outs[0], 1], time[in_outs[0]], s=50, color='green')
        # ax.scatter(tracking[in_outs[1], 0], tracking[in_outs[1], 1], time[in_outs[1]], s=50, color='blue')

        # Plt projections
        ax.plot(tracking[:, 0], tracking[:, 1], ones, color=[.7, .7, .7], zorder=-1)
        ax.plot(ones, tracking[:, 1], time, color=[.85, .85, .85], zorder=-1)
        ax.plot(tracking[:, 0], ones*-20, time, color=[.85, .85, .85], zorder=-1)

        # ax.set_axis_off()
        ax.view_init(elev=15, azim=-80)

        if not individual:
            _ = ax.set(title=e.mouse_id, zticks=[], xticks=[], 
                            yticks=[], xlim=[-20, 1000])
        else:
            _ = ax.set(title=e.mouse_id, zticks=[], xticks=[], 
                        xlabel='X', ylabel='Y', zlabel='TIME',
                            yticks=[], xlim=[-20, 1000])
            
            save_figure(fig, os.path.join(maze_fld, f"mouse - {e.mouse_id}"))

    if not individual:
        save_figure(fig, os.path.join(outfld, f"explorations maze {maze}"))


    break


# %%
"""
    Look at the avg duration of T-S and S-T trips per maze during explorations
"""

PLOT_TRACKING = False # se to True to check that shelter and threat in/outs are correctly detected

skip_first_n_frames = 3000

trips = {f'maze{m}':{'st':{'l':[], 'r':[]}, 'ts':{'l':[], 'r':[]}} for m in mazes}

for n, maze in enumerate(mazes):
    explorations = get_maze_explorations(maze,  naive=None, lights=1)

    if PLOT_TRACKING:
        f, axarr = plt.subplots(figsize=(20, 20), ncols=7, nrows=7)
        axarr = axarr.flatten()

    for i,e in tqdm(explorations.iterrows()):
        # Get tracking
        tracking = e.body_tracking
        x, y = tracking[:, 0], tracking[:, 1]
        time = np.arange(len(tracking))

        # make custom in roi
        roi = np.zeros_like(time)
        roi[(x>440)&(x<550)&(y>590)] = 1 # in shelter
        roi[(x>440)&(x<550)&(y<350)] = -1 # in threat

        in_shelt = np.zeros(len(time))
        in_shelt[roi == 1] = 1

        in_threat = np.zeros(len(time))
        in_threat[roi == -1] = 1

        tracking = tracking[skip_first_n_frames:, :]
        in_shelt = in_shelt[skip_first_n_frames:]
        in_threat = in_threat[skip_first_n_frames:]

        # Get first time the mouse reaches the shelter
        in_outs = get_times_signal_high_and_low(in_shelt, th=.5)
        if len(in_outs[0]):
            first_shelt = in_outs[0][0]
        else:
            print('cont')
            continue

        tracking = tracking[first_shelt:]
        in_shelt = in_shelt[first_shelt:]
        in_threat = in_threat[first_shelt:]

        if len(tracking) != len(in_shelt) or len(in_shelt) != len(in_threat):
            raise ValueError

        # Get shelter and threat enter/exits
        shelt_in_outs = get_times_signal_high_and_low(in_shelt, th=.5)
        threat_in_outs = get_times_signal_high_and_low(in_threat, th=.5)



        # Get S -> T trips
        st_trips = []
        for sexit in shelt_in_outs[1]:
            next_at_t = [t for t in threat_in_outs[0] if t > sexit]
            if not next_at_t: 
                continue

            # Check if there's a shelter enter before the threat enter: incomplete trip
            next_sexit = [s for s in shelt_in_outs[0] if s > sexit and s < next_at_t[0]]
            if next_sexit: continue

            st_trips.append((sexit, next_at_t[0]))


        # Get T -> s trips
        ts_trips = []
        for texit in threat_in_outs[1]:
            next_at_s = [s for s in shelt_in_outs[0] if s > texit]
            if not next_at_s: 
                continue

            # Check if there's a threat enter before the shelter enter: incomplete trip
            next_tenter = [t for t in threat_in_outs[0] if t > texit and t < next_at_s[0]]
            if next_tenter: continue

            ts_trips.append((texit, next_at_s[0]))
      

        # Plot trips tracking
        if PLOT_TRACKING:
            for start, end in st_trips:
                axarr[i].scatter(tracking[start:end, 0], tracking[start:end, 1]+500, c=np.arange(end-start))

            for start, end in ts_trips:
                axarr[i].scatter(tracking[start:end, 0], tracking[start:end, 1], c=np.arange(end-start))

        # add trips lengths to dictionary
        trips[f'maze{maze}']['st']['l'].extend([(end-start)/e.fps for start, end in st_trips if
                                         np.mean(tracking[start:end, 0])<500])
        trips[f'maze{maze}']['st']['r'].extend([(end-start)/e.fps for start, end in st_trips if
                                         np.mean(tracking[start:end, 0])>=500])

        trips[f'maze{maze}']['ts']['l'].extend([(end-start)/e.fps for start, end in ts_trips if
                                         np.mean(tracking[start:end, 0])<500])
        trips[f'maze{maze}']['ts']['r'].extend([(end-start)/e.fps for start, end in ts_trips if
                                         np.mean(tracking[start:end, 0])>=500])
# %%

f, axarr = plt.subplots(figsize=(16, 12), nrows=2, sharex=True )

for n, (maze, trps) in enumerate(trips.items()):
    st_l = np.mean(trps['st']['l'])
    st_r = np.mean(trps['st']['r'])
    ts_l = np.mean(trps['ts']['l'])
    ts_r = np.mean(trps['ts']['r'])

    std_st_l = np.std(trps['st']['l'])
    std_st_r = np.std(trps['st']['r'])
    std_ts_l = np.std(trps['ts']['l'])
    std_ts_r = np.std(trps['ts']['r'])

    x = np.array([n-.15, n+.15])
    axarr[0].bar(x, [st_l, st_r], color=paper.maze_colors[maze], yerr=[std_st_l, std_st_r], width=.2, capsize =9, label='shelter -> threat')

    axarr[0].errorbar(x+0.05, [ts_l, ts_r], color=desaturate_color(paper.maze_colors[maze]), yerr=[std_ts_l, std_ts_r],
            ecolor=[.4, .4, .4], elinewidth=3)
    axarr[0].scatter(x+0.05, [ts_l, ts_r], color=desaturate_color(paper.maze_colors[maze]),s=250, zorder=99, ec='k', label='threat -> shelter')

    
    # n trips per arm
    n_st_l = len(trps['st']['l'])
    n_st_r = len(trps['st']['r'])
    n_ts_l = len(trps['ts']['l'])
    n_ts_r = len(trps['ts']['r'])


    axarr[1].bar(x, [n_st_l, n_st_r], color=paper.maze_colors[maze], width=.2, capsize =9, label='shelter -> threat')
    axarr[1].scatter(x, [n_ts_l, n_ts_r], color=desaturate_color(paper.maze_colors[maze]),s=250, zorder=99, ec='k', label='threat -> shelter')

    if n == 0: 
        axarr[0].legend()
        axarr[1].legend()


_ =axarr[0].set(title='Average trip duration', ylabel='duration per arm (s)',  xticklabels=[f'maze{m}' for m in mazes], xticks=np.arange(5))
_ =axarr[1].set(title='Number of trips per arm', ylabel='# trips per arm',  xticklabels=[f'maze{m}' for m in mazes], xticks=np.arange(5))

clean_axes(f)
save_figure(f, os.path.join(plots_dir, f"trips per path per maze"))