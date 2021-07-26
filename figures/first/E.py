# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.plot.distributions import plot_distribution


from figures._plot_utils import generate_figure, triple_plot, plot_trial_tracking
from figures.colors import tracking_color, start_color, end_color
from figures.first import M6, M4, fig_1_path
from figures.settings import trace_downsample, dpi
from figures.bayes import Bayes

'''
    Plot tracking for all trials in M6

    Plot posteriors for M4 vs M6
'''
print(M6)

# %%
# plot tracking data
ax = generate_figure()

for n, trial in M6.trials.iterrows():
    plot_trial_tracking(ax, trial, tracking_color, start_color, end_color, downsample=trace_downsample)
ax.axis('off')
ax.set(title=M6.name)
ax.figure.savefig(fig_1_path / 'panel_E_tracking.eps', format='eps', dpi=dpi)
# %%
# plot posteriors

ax = generate_figure()
bayes = Bayes()
for n, data in enumerate((M4, M6)):
    a, b, _, _, _, _, _ = bayes.grouped_bayes_analytical(data.n_trials, data.nR)
    plot_distribution(a, b, dist_type='beta', ax=ax, plot_kwargs=dict(color=data.color, label=data.name), shaded=True)

    triple_plot(
        -n*1.4 - 1.4, 
        data.mice_pR(),
        ax, 
        kde_kwargs=dict(bw=0.05),
        kde_normto=.4,
        box_width=.2,
        color=data.color,
        fill=.001,
        horizontal=True,
        spread=0.02)

ax.axhline(0, lw=2, color='k')
ax.plot([0.5, 0.5], [0, 9], ls='--', lw=2, color=[.4, .4, .4], zorder=-1)
ax.legend()
ax.set(ylabel='density', xlabel='p(R)', xlim=[-0.02, 1.02])
ax.figure.savefig(fig_1_path / 'panel_E_posteriors.eps', format='eps', dpi=dpi)


# %%
from rich import print
from scipy.stats import fisher_exact
table = [
    [
        M4.nR, M4.n_trials, 
    ],
    [
        M6.nR, M6.n_trials
    ]]

from scipy.stats import chisquare

print(chisquare(
    [M4.nR, M6.nR],
    [0.5 * M4.n_trials, 0.5 * M6.n_trials]
))


# _, pval = fisher_exact(table)
# if pval < 0.05:
#     print(f'The probability of reaching the shelter is [green]different[/green] between the two conditions with p value: {pval}')
# else:
#     print(f'The probability of reaching the shelter is [red]NOT different[/red] between the two conditions with p value: {pval}')
# %%