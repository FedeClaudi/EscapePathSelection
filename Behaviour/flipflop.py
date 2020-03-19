# %%
import sys
sys.path.append('./')
from Utilities.imports import *

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
palette = makePalette(teal, lilla, 4, False)
arms_colors = {
    "left": teal,
    'center': darkcyan,
    "right": lilla,
}


fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\flipflop"

ea = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea.add_condition("m1", maze_design=1, lights=1, escapes_dur=True, tracking="all"); print("Got m1")

# %%
# TODO then make metadata to know the state of the maze at each trial, split the data and get p(R) grouped and individuals
# TODO then find a way to extract the maze state from the video and look at the behaviour immediately after the flip and compare with baseline exploration


# %%
# ------------------------ GET STIMULI FOR EACH MOUSE ------------------------ #
getmetadata = False
if getmetadata:
    try:
        stims = pd.DataFrame((Stimuli * Session * Trials * Trials.TrialSessionMetadata.proj(ename='experiment_name', session_frame="stim_frame_session") & "experiment_name='FlipFlop Maze'").fetch())
    except: 
        pass
    sessions = set(stims.uid.values)

    txt = open(os.path.join(fld, "trials_data_1.txt"), "a")

    store = {}
    for n, sess in enumerate(sessions):
        if sess < 184: fps= 30
        else: fps = 40
        trials = stims.loc[stims.uid == sess].sort_values(by=['session_frame'])
        
        date = trials.date.values[0]
        mouse = trials.mouse_id.values[0]

        sstore = []
        txt.write("\n\nSession {} of {}: {} - {} - {}".format(n, len(sessions), sess, date, mouse))
        for i, t in trials.iterrows():
            frame = t.session_frame
            if frame < 0 : continue
            txt.write("\n     trial {}. Frame {}, min {}, arm {}".format(t.stimulus_uid, frame, round(frame / fps / 60, 2), t.escape_arm))
            sstore.append(str(t.stimulus_uid))
        store[int(sess)] = sstore

    with open(os.path.join(fld, "trials_metadata_template_1.yml"), "w") as f:
        yaml.dump(store, f)
    txt.close()


# %%
# ---------------------------------------------------------------------------- #
#                               GET DATA COMPLETE                              #
# ---------------------------------------------------------------------------- #
max_duration_th = 9


trials1 = pd.DataFrame((Session * Trials * Trials.TrialTracking * Trials.TrialSessionMetadata.proj(ename='experiment_name', session_frame="stim_frame_session")\
						& "escape_duration > 0" & "experiment_name='FlipFlop2 Maze'").fetch())
trials2 = pd.DataFrame((Session * Trials * Trials.TrialTracking * Trials.TrialSessionMetadata.proj(ename='experiment_name', session_frame="stim_frame_session")\
						& "escape_duration > 0" & "experiment_name='FlipFlop Maze'").fetch())

trials = pd.concat([trials1, trials2])

trials = trials.loc[trials.escape_duration <= max_duration_th]
trials = trials.loc[trials.escape_duration > 0]

metadata = load_yaml(os.path.join(fld, "trials_metadata.yml"))
maze_state = []
for i, t in trials.iterrows():
    maze_state.append([v for l in metadata[int(t.uid)]  for k, v in l.items() if k==t.stimulus_uid][0])
trials['maze_state'] = maze_state

left_long_trials = trials.loc[trials.maze_state == 'L']
right_long_trials = trials.loc[trials.maze_state == 'R']

for side, tr in zip(['LEFT', 'RIGHT'], [left_long_trials, right_long_trials]):
    l_esc = len(tr.loc[tr.escape_arm == 'left'])
    r_esc = len(tr.loc[tr.escape_arm == 'right'])
    pr = round(r_esc/len(tr), 3)

    print("\n{} long: {} trials, {}R - {}L, p(R):{}".format(side, len(tr), r_esc, l_esc, pr))

# plot tracking
plot_tracking = True
if plot_tracking:
    f, axarr = create_figure(subplots=True, ncols=2)

    for i, trial in left_long_trials.iterrows():
        axarr[0].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=arms_colors[trial.escape_arm], alpha=.8)
    for i, trial in right_long_trials.iterrows():
        axarr[1].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=arms_colors[trial.escape_arm], alpha=.8)

    axarr[0].set(title="LEFT LONG", xlim=[0, 1000], ylim=[0, 1000])
    axarr[1].set(title="RIGHT LONG", xlim=[0, 1000], ylim=[0, 1000])


# %%
# --------------------------------- PLOT P(R) -------------------------------- #

pRs = ea.bayes_by_condition_analytical()


n_ll, k_ll = len(left_long_trials), len(left_long_trials.loc[left_long_trials.escape_arm == 'right'])
n_rl, k_rl = len(right_long_trials), len(right_long_trials.loc[right_long_trials.escape_arm == 'right'])

a_ll, b_ll, mean_ll, mode, sigmasquared_ll, prange = ea.grouped_bayes_analytical(n_ll, k_ll)
a_rl, b_rl, mean_rl, mode, sigmasquared_rl, prange = ea.grouped_bayes_analytical(n_rl, n_rl-k_rl)

f, axarr = create_figure(subplots=True, ncols=2, figsize=(10, 10), sharey=True)


# PLOT subsample from M1
bootstrap = False
if bootstrap:
    for i in range(50):
        n = 41 
        trials = ea.conditions['m1'].sample(n) 
        k = len(trials.loc[trials.escape_arm == 'right'])

        a, b, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical(n, k)
        plot_distribution(a, b, ax=axarr[0], dist_type="beta", shaded=True, line_alpha=0, shade_alpha=.02, plot_kwargs={"color":black})

    for i in range(50):
        n = 91 
        trials = ea.conditions['m1'].sample(n) 
        k = len(trials.loc[trials.escape_arm == 'right'])

        a, b, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical(n, k)
        plot_distribution(a, b, ax=axarr[0], dist_type="beta", shaded=True, line_alpha=0, shade_alpha=.02, plot_kwargs={"color":black})

# PLOT M1
pr = pRs.loc[pRs.condition == 'm1']
axarr[0].errorbar(pr['mean'], -.75, xerr=2*math.sqrt(pr['sigmasquared']), color=darksalmon)
axarr[0].scatter(pr['mean'], -.75, color=darksalmon, s=250, edgecolor=black, zorder=99, label="M1 n:{}".format(len(ea.conditions['m1'])))
plot_distribution(pr['alpha'].values[0], pr['beta'].values[0], ax=axarr[0], dist_type="beta", shaded="True", line_alpha=.6,
                plot_kwargs={"color":darksalmon}, shade_alpha=.1, )

# PLOT AFTER FLIP
axarr[0].errorbar(mean_rl, -.25, xerr=2*math.sqrt(sigmasquared_rl), color=goldenrod)
axarr[0].scatter(mean_rl,-.25, color=goldenrod, s=250, edgecolor=black, zorder=99, label="afterflip n:{}".format(n_rl))
plot_distribution(a_rl, b_rl, ax=axarr[0], dist_type="beta", shaded="True", line_alpha=.8,
                plot_kwargs={"color":goldenrod}, shade_alpha=.2, )

# PLOT BASELINE
axarr[0].errorbar(mean_ll, -.5, xerr=2*math.sqrt(sigmasquared_ll), color=darkseagreen)
axarr[0].scatter(mean_ll, -.5, color=darkseagreen, s=250, edgecolor=black, zorder=99, label="baseline n:{}".format(n_ll))
plot_distribution(a_ll, b_ll, ax=axarr[0], dist_type="beta", shaded="True", line_alpha=.8,
                plot_kwargs={"color":darkseagreen}, shade_alpha=.2 )


axarr[0].plot([0.5, .5], [0, 12], ls="--", lw=3, color=black, alpha=.5)
axarr[0].axhline(0, lw=3, color=black, alpha=.5)

axarr[0].legend()
_ = axarr[0].set(title="p(shortest) before and after flip", xlabel="p(shortest)", ylabel="density")


# Plot the difference between m1 and after flip
m1samples = random.choices(get_distribution('beta', pr['alpha'].values[0], pr['beta'].values[0], n_samples=500000), k=100000)
afterflipsamples = random.choices(get_distribution('beta',a_rl, b_rl, n_samples=500000), k=100000)
diff =  np.array(afterflipsamples) - np.array(m1samples) 

axarr[1].axvline(0, ls="--", lw=4, color=black, alpha=.6)
_ = axarr[1].hist(diff, bins=50, color=goldenrod, alpha=.5, histtype="stepfilled", density=True, zorder=80)
_ = axarr[1].hist(diff, bins=50, color=goldenrod, alpha=1, histtype="step", lw=4, density=True, zorder=85, label="after flip")

mean, std = np.mean(diff), np.std(diff)
axarr[1].errorbar(mean, -.3, xerr=std, color=goldenrod, lw=4, zorder=98)
axarr[1].scatter(mean, -.3, color=goldenrod, edgecolor=black, s=250, lw=2, zorder=99)


beforeflip = random.choices(get_distribution('beta',a_ll, b_ll, n_samples=500000), k=100000)
diff = np.array(beforeflip) - np.array(m1samples)

_ = axarr[1].hist(diff, bins=50, color=darkseagreen, alpha=.5, histtype="stepfilled", density=True, zorder=80)
_ = axarr[1].hist(diff, bins=50, color=darkseagreen, alpha=1, histtype="step", lw=4, density=True, zorder=85, label="baseline")
axarr[1].axhline(0, lw=4, color=black, alpha=.6, zorder=99)

mean, std = np.mean(diff), np.std(diff)
axarr[1].errorbar(mean, -.8, xerr=std, color=darkseagreen, lw=4, zorder=98)
axarr[1].scatter(mean, -.8, color=darkseagreen, edgecolor=black, s=250, lw=2, zorder=99)

axarr[1].legend()
_ = axarr[1].set(xlabel='difference', ylabel="density", title="p(R|m1) - p(R|flipflop)")

# %%
# ----------------------------------- TESET ---------------------------------- #

escapes =  [1 if i.escape_arm == 'right' else 0 for _, i in ea.conditions['m1'].iterrows()]

prs, prs2, prs3 = [], [], []
for i in tqdm(range(10000)):
    prs.append(np.mean(random.choices(escapes, k=41)))
    prs2.append(np.mean(random.choices(escapes, k=91)))

_  = plt.hist(prs, bins=10, density=True, alpha=.5)
_  = plt.hist(prs2, bins=10, density=True, alpha=.5)


