# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from figures._plot_utils import generate_figure
from figures.colors import tracking_color_dark
from figures.first import M1, M2, M3, fig_1_path
from figures.settings import trace_downsample, dpi

# %%

axes = generate_figure(ncols=3, figsize=(18, 6))

for ax, ds in zip(axes, (M1, M2, M3)):

    for n, trial in ds.trials.iterrows():
        ax.plot(trial.x[::trace_downsample], trial.y[::trace_downsample], 
                color=tracking_color_dark, alpha=.4)
    ax.axis('off')
    ax.set(title=ds.name)
ax.figure.savefig(fig_1_path / 'panel_C.eps', format='eps', dpi=dpi)
# %%
