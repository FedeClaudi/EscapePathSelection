
# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from figures._plot_utils import generate_figure, plot_trial_tracking
from figures.colors import tracking_color, start_color, end_color
from figures.first import M4, fig_1_path
from figures.settings import trace_downsample, dpi

'''
    Plot tracking from all trials in M4
'''

# %%

ax = generate_figure()
for n, trial in M4.trials.iterrows():
    if 88 < trial.uid < 100: 
        plot_trial_tracking(ax, trial, tracking_color, start_color, end_color, downsample=trace_downsample)
ax.axis('off')
ax.figure.savefig(fig_1_path / 'panel_A_central.eps', format='eps', dpi=dpi)
# %%

ax = generate_figure()
for n, trial in M4.trials.iterrows():
    if 88 < trial.uid < 100: continue
    plot_trial_tracking(ax, trial, tracking_color, start_color, end_color, downsample=trace_downsample)
ax.axis('off')
ax.figure.savefig(fig_1_path / 'panel_A_no_central.eps', format='eps', dpi=dpi)
# %%
# get the ToT
_trials = M4.trials.loc[(M4.trials.uid < 88)|(M4.trials.uid > 100)]
print(_trials.time_out_of_t.mean(), _trials.time_out_of_t.std())
# %%
