import matplotlib.pyplot as plt
import numpy as np

from fcutils.plot.figure import clean_axes, set_figure_subplots_aspect

def generate_figure(flatten=True, aspect_kwargs={}, **kwargs):
    figsize = kwargs.pop("figsize", (9, 9))
    f, axes = plt.subplots(figsize=figsize, **kwargs)
    clean_axes(f)

    if isinstance(axes, np.ndarray) and flatten:
        axes = axes.flatten()

    if aspect_kwargs:
        set_figure_subplots_aspect(**aspect_kwargs)

    return axes