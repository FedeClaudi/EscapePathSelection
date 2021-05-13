
# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from figures._plot_utils import generate_figure
from figures.colors import tracking_color_dark
from figures.first import M4, fig_1_path
from figures.settings import trace_downsample, dpi

# %%

ax = generate_figure()
for n, trial in M4.trials.iterrows():
    ax.plot(trial.x[::trace_downsample], trial.y[::trace_downsample], color=tracking_color_dark, alpha=.6)
ax.axis('off')
ax.figure.savefig(fig_1_path / 'panel_A.eps', format='eps', dpi=dpi)
# %%
