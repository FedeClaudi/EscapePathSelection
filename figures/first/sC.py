# %%
import sys
from pathlib import Path
import os
import numpy as np
from scipy.stats import sem

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from myterial import green

from fcutils.maths.fit import linear_regression

from figures.first import M1, M2, M3, M4, M6, fig_1_path
from figures._plot_utils import generate_figure, triple_plot
from figures.colors import tracking_color
from figures.settings import dpi, max_escape_duration

print(M1, M2, M3, M4, sep='\n\n')

# %%
# correlation between escape duration and distance travelled
ax = generate_figure()

X, Y, C = [], [], []
for dataset in (M1, M2, M3, M4):
    for i,trial in dataset.iterrows():
        if trial.escape_arm == 'center': continue

        distance_travelled = np.sum(np.sqrt(
            np.diff(trial.x)**2 + np.diff(trial.y)**2
        ))
        X.append(distance_travelled) 
        Y.append(trial.escape_duration)
        C.append(tracking_color if trial.escape_arm=='right' or dataset.name=='M4' else green)

# fit linear regression and plot
ax.scatter(X, Y, s=50, c=C)
_, intercept, slope, _ = linear_regression(X, Y, robust=False)

x0, x1 = 80, 250
ax.plot(
    [x0, x1],
    [slope * x0 + intercept, slope*x1 + intercept], 
    lw=2,
    color=[.2, .2, .2]
)

ax.axhline(max_escape_duration, ls=':', color=[.4, .4, .4])

_ = ax.set(title='Distance vs escape duration', ylabel='escape duration\n$s$', xlabel='distance travelled\n$cm$')
ax.figure.savefig(fig_1_path / 'panel_S_C_dist_time_correlation.eps', format='eps', dpi=dpi)

# %%
# plot left vs right duration
ax = generate_figure()

datasets = (M1, M2, M3, M4, M6)
shift = .6
for n, dataset in enumerate(datasets):
    n = n*3
    L_dur = dataset.L.escape_duration.values
    R_dur = dataset.R.escape_duration.values

    # ax.bar([n-0.25, n+0.25], [L_dur.mean(), R_dur.mean()], width=0.5, color=[dataset.color, dataset.mice_color], yerr= [sem(L_dur), sem(R_dur)])

    triple_plot(
                n - shift, 
                L_dur, 
                ax, 
                shift=0.3, 
                scatter_kws=dict(s=20), 
                box_width=0.2,
                kde_normto=0.4,
                pad=.4, 
                invert_order=True,
                color=dataset.color
                )

    triple_plot(
                n + shift, 
                R_dur, 
                ax,
                shift=0.3, 
                scatter_kws=dict(s=20), 
                box_width=0.2,
                kde_normto=0.4,
                pad=0.4,
                color=dataset.mice_color)

    
_ = ax.set(xticks=np.arange(5)*3+shift, xticklabels=[ds.name for ds in datasets], ylabel='duration \n (s)', xlabel='maze')
ax.figure.savefig(fig_1_path / 'panel_S_C_duration_per_arm.eps', format='eps', dpi=dpi)


# %%
