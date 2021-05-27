# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.maths.functions import centered_logistic
from fcutils.plot.distributions import plot_fitted_curve, plot_distribution

from figures.first import M1, M2, M3, M4, fig_1_path
from figures._plot_utils import generate_figure, triple_plot
from figures.statistical_tests import fisher
from figures.bayes import Bayes
from figures.settings import dpi

print(M1, M2, M3, M4, sep='\n\n')


'''
    Plot psychometric and posteriors for M1 to M4
'''


# %% get

# ---------------------------------------------------------------------------- #
#                                   M1 TO M4                                   #
# ---------------------------------------------------------------------------- #

# for M1 and M4 check if p(R) != 0.5
print(f'''
    Escape probabilities by arm:
        M1 {M1.escape_probability_by_arm()}
        M2 {M2.escape_probability_by_arm()}
        M3 {M3.escape_probability_by_arm()}
        M4 {M4.escape_probability_by_arm()}
''')

table = np.array([
    list(M1.escape_numbers_by_arm().values()),
    list(M4.escape_numbers_by_arm().values())
])


fisher(table, ' p(R) in M4 vs M1')

# %%

# -------------- plot psychometric curve across maze 1->4 (+ 6) -------------- #
bayes = Bayes()

axes = generate_figure(ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(16, 9))

datasets = [M4, M3, M2, M1]

X, Y, YERR = [], [], []
for n, data in enumerate(datasets):
    # Plot global mean for all trials
    prange = data.grouped_pR()
    X.extend([data.maze['ratio'] for _ in data.mice_pR()])
    Y.extend(data.mice_pR())
    # YERR.append(prange.sem)

    axes[0].scatter(data.maze['ratio'], prange.mean, s=200, edgecolors=data.color, zorder=200, color='w')


    triple_plot(
        data.maze['ratio'],
        data.mice_pR(),
        axes[0],
        color=data.color, -+*+*
        shift=0.005, 
        zorder=100, 
        scatter_kws=dict(s=50), 
        kde_kwargs=dict(bw=0.05),
        box_width=0.005,
        kde_normto=.02,
        fill=0.005, 
        pad=0.0,
        spread=0.001,
    )

    # plot bayesian posterior
    a, b, _, _, _, _, _ = bayes.grouped_bayes_analytical(data.n_trials, data.nR)
    plot_distribution(a, b, dist_type='beta', ax=axes[1], shaded=True, vertical=True, plot_kwargs=dict(color=data.color))


# plot psychometric
curve_params = plot_fitted_curve(
                centered_logistic,
                X,
                Y,
                axes[0],
                xrange=[.4, .7], 
                scatter_kwargs=dict(alpha=0),
                # fit_kwargs = dict(sigma=YERR),
                line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))

# fix axes
# ratios = [data.maze['ratio'] for data in datasets]
axes[0].legend()
_ = axes[0].set(title='Psychometric curve', ylabel='$p(R)$', xlabel='maze geodesic ratio', ylim=[-0.02, 1.02])
_ = axes[1].axis('off')
axes[1].figure.savefig(fig_1_path / 'panel_D.eps', format='eps', dpi=dpi)

# %%

