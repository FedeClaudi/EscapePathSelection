import sys
sys.path.append('./')

import numpy as np

from fcutils.file_io.io import load_yaml
from fcutils.maths.geometry import calc_distance_from_point

def get_mazes():
    if sys.platform == 'darwin':
        maze_metadata_file = '/Users/federicoclaudi/Documents/Github/EscapePathSelection/paper/dbase/Mazes_metadata.yml'
    else:
        maze_metadata_file = "C:\\Users\\Federico\\Documents\\GitHub\\EscapePathSelection\\paper\\dbase\\Mazes_metadata.yml"
    mazes = load_yaml(maze_metadata_file)

    for maze, metadata in mazes.items():
        mazes[maze]['ratio'] = metadata['left_path_length']/(metadata['right_path_length'] + metadata['left_path_length'])
    return mazes

def get_euclidean_dist_for_dataset(datasets, shelter_pos):
    eucl_dists, eucl_dists_means, eucl_dists_traces = {}, {}, {}
    for name, data in datasets.items():

        means = {a:[] for a in ['left', 'right']}
        traces = {a:[] for a in ['left', 'right']}
        
        for n, trial in data.iterrows():
            if trial.escape_arm == "center": continue

            d = calc_distance_from_point(trial.body_xy[trial.out_of_t_frame-trial.stim_frame:, :], shelter_pos)
            means[trial.escape_arm].append(np.mean(d))
            traces[trial.escape_arm].append(d)

        eucl_dists[name] = np.nanmean(means['left'])/(np.nanmean(means['right']) + np.nanmean(means['left']))
        eucl_dists_means[name] = means
        eucl_dists_traces[name] = traces
    return eucl_dists, eucl_dists_means, eucl_dists_traces

if __name__ == '__main__':
    get_mazes()

# -------------------------------- Path length ------------------------------- #


# add_m0 = False

# mazes = load_yaml("database/maze_components/Mazes_metadata.yml")
# f, ax = create_figure(subplots=False)

# ax.plot([0, 10000], [0, 10000], ls=':', lw=2, color=[.2, .2, .2], alpha=.3)

# for maze, metadata in mazes.items():
#         ax.scatter(metadata['right_path_length'], metadata['left_path_length'], color=maze_colors[maze], edgecolor=black, s=250, zorder=99)
#         _ = vline_to_point(ax, metadata['right_path_length'], metadata['left_path_length'], color=maze_colors[maze], ymin=0, ls="--", alpha=.5, zorder=0)
#         _ = hline_to_point(ax, metadata['right_path_length'], metadata['left_path_length'], color=maze_colors[maze], ls="--", alpha=.5, zorder=0)

# _ = ax.set(title="Path lengths", xlabel='length of shortest', ylabel='length of longest', xlim=[400, 1150], ylim=[400, 1150])
# save_plot("path_lengths", f)


# ---------------------------- Euclidean distance ---------------------------- #


# plot_single_trials = True

# euclidean_dists = {}
# f, ax = create_figure(subplots=False)
# xticks, xlabels = [], []
# for i, (condition, trials) in enumerate(ea.conditions.items()):
#     # Get data
#     if condition not in five_mazes: continue

#     means, maxes = {a:[] for a in ['left', 'right']}, {a:[] for a in ['left', 'right']}
#     for n, trial in trials.iterrows():
#         if trial.escape_arm == "center": continue

#         d = calc_distance_from_shelter(trial.body_xy[trial.out_of_t_frame-trial.stim_frame:, :], [500, 850])
#         means[trial.escape_arm].append(np.mean(d))
#         maxes[trial.escape_arm].append(np.max(d))

    

#     # Make plot
#     x = [i-.25, i+.25]
#     xticks.extend([i-.25,i, i+.25])
#     xlabels.extend(["left","\n{}".format(condition), "right"])
#     ax.axvline(i-.25, color=[.2, .2, .2], ls=":", alpha=.15)
#     ax.axvline(i+.25, color=[.2, .2, .2], ls=":", alpha=.15)

#     y = [np.mean(means['left']), np.mean(means['right'])]
#     yerr = [stats.sem(means['left']), stats.sem(means['right'])]
#     ax.plot(x, y, "-o", label=condition, color=maze_colors[condition], zorder=90)
#     ax.scatter(x, y, edgecolor=black, s=250, color=maze_colors[condition], zorder=99)
#     ax.errorbar(x, y, yerr, color=maze_colors[condition], zorder=90)


#     ttest, pval = stats.ttest_ind(means['left'], means['right'])
#     if pval < .05:
#         ax.plot([i-.3, i+.3], [505, 505], lw=4, color=[.4, .4, .4])
#         ax.text(i-0.025, 505, "*", fontsize=20)
#     else:
#         ax.plot([i-.3, i+.3], [505, 505], lw=4, color=[.7, .7, .7])
#         ax.text(i-0.05, 508, "n.s.", fontsize=16)

#     # Take average and save it
#     euclidean_dists[condition] = y[0]/y[1]

# _ = ax.set(title="Average euclidean distance", xticks=xticks, xticklabels=xlabels,
#                 ylabel="mean distance (s)")
# save_plot("euclidean_dist", f)





# ------------------------------ Escape duration ----------------------------- #

# path_durations, alldurations = ea.get_duration_per_arm_from_trials()

# xticks, xlabels = [], []
# f, ax = create_figure(subplots=False)
# for i, (condition, durations) in enumerate(path_durations.items()):
#     if condition not in five_mazes: continue
#     x = [i-.25, i+.25]
#     xticks.extend([i-.25,i, i+.25])
#     xlabels.extend(["left","\n{}".format(condition), "right"])
#     y = [durations.left.mean, durations.right.mean]
#     yerr = [durations.left.sem, durations.right.sem]

#     print("Maze {} - ratio: {}".format(condition, round(y[0]/y[1], 2)))

#     ax.axvline(i-.25, color=[.2, .2, .2], ls=":", alpha=.15)
#     ax.axvline(i+.25, color=[.2, .2, .2], ls=":", alpha=.15)

#     ax.plot(x, y, "-o", label=condition, color=maze_colors[condition], zorder=90)
#     ax.scatter(x, y, edgecolor=black, s=250, color=maze_colors[condition], zorder=99)
#     ax.errorbar(x, y, yerr, color=maze_colors[condition], zorder=90)

#     ttest, pval = stats.ttest_ind(alldurations[condition]['left'], alldurations[condition]['right'])
#     if pval < .05:
#         ax.plot([i-.3, i+.3], [4.75, 4.75], lw=4, color=[.4, .4, .4])
#         ax.text(i-0.025, 4.75, "*", fontsize=20)
#     else:
#         ax.plot([i-.3, i+.3], [4.75, 4.75], lw=4, color=[.7, .7, .7])
#         ax.text(i-0.05, 4.8, "n.s.", fontsize=16)
        

# _ = ax.set(title="Paths durations per maze design", xticks=xticks, xticklabels=xlabels,
#                 ylabel="mean duration (s)")
# save_plot("escape_durations", f)





# ---------------------------- DURATION VS LENGTH ---------------------------- #


# f, ax = create_figure(subplots=False)
# durs, dists, speeds = [], [], []
# for condition, trials in ea.conditions.items():
#     if condition in ["m0", "m6"]: continue
#     for n, (i, trial) in enumerate(trials.iterrows()):
#         dist = np.sum(calc_distance_between_points_in_a_vector_2d(trial.body_xy[trial.out_of_t_frame-trial.stim_frame:,  :]))
#         dur = (trial.at_shelter_frame - trial.out_of_t_frame)/trial.fps
#         durs.append(dur); dists.append(dist); speeds.append(np.mean(trial.body_speed[trial.out_of_t_frame-trial.stim_frame:]))

# ax.scatter(dists, durs, c=speeds, cmap="inferno_r", alpha=.7)
# ax.set(xlabel="Distance", ylabel="Duration")
# save_plot("dur_vs_dist", f)


