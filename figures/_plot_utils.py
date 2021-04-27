import matplotlib.pyplot as plt
import numpy as np

from fcutils.plot.figure import clean_axes, set_figure_subplots_aspect

from figures.settings import max_escape_frames, max_escape_duration, fps

# parameters to style axes with time from escape onset on X
time_xax_params = dict(
    xlabel='time', 
    xticks=np.arange(0, max_escape_frames, fps * 2),
    xticklabels=np.arange(0, max_escape_duration, 2),
)

def generate_figure(flatten=True, aspect_kwargs={}, **kwargs):
    figsize = kwargs.pop("figsize", (9, 9))
    f, axes = plt.subplots(figsize=figsize, **kwargs)
    clean_axes(f)

    if isinstance(axes, np.ndarray) and flatten:
        axes = axes.flatten()

    if aspect_kwargs:
        set_figure_subplots_aspect(**aspect_kwargs)

    return axes