import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import circmean, circstd
from pointgrid import align_points_to_grid

from myterial import salmon

from fcutils.plot.distributions import plot_kde
from fcutils.plot.figure import clean_axes, set_figure_subplots_aspect
from fcutils.plot.elements import plot_mean_and_error

from figures.settings import max_escape_frames, max_escape_duration, fps
from figures.colors import tracking_color, tracking_color_dark

# parameters to style axes with time from escape onset on X
time_xax_params = dict(
    xlabel='time', 
    xticks=np.arange(0, max_escape_frames, fps * 2),
    xticklabels=np.arange(0, max_escape_duration, 2),
)


def triple_plot(
        x_pos, 
        y, ax, 
        color='k', 
        shift=0.25, 
        zorder=100, 
        scatter_kws=None, 
        kde_kwargs=None,
        box_width=0.5,
        kde_normto=None,
        invert_order = False,
        fill=0.1, 
        pad=0.0,
        spread=0.01,
        horizontal=False
    ):
    '''
        Given a 1d array of data it plots a scatter of the data (with x_pos as X coords)
        a box plot of the distribution and a KDE of the distribution
    '''
    if invert_order:
        scatter_x = 2 * shift
        box_x = shift
        kde_x = 0
        if kde_normto is not None:
            kde_normto = - kde_normto
    else:
        scatter_x = 0
        box_x = shift
        kde_x = 2 * shift
    x = np.random.normal(x_pos, spread, size=len(y))

    # scatter plot
    data = align_points_to_grid(np.vstack([x, y]).T, fill=fill, pad=pad)
    scatter_kws = scatter_kws or dict(s=15)
    if horizontal:
        ax.scatter(data[:, 1] + scatter_x, data[:, 0], color=color, **scatter_kws, zorder=zorder+1)
    else:
        ax.scatter(data[:, 0] + scatter_x, data[:, 1], color=color, **scatter_kws, zorder=zorder+1)

    # box plot
    boxes = ax.boxplot(
        y, 
        positions=[x_pos+box_x], 
        zorder=zorder, 
        widths=box_width,
        showcaps=False,
        showfliers=False,
        patch_artist=True,
        boxprops = dict(color='k'),
        whiskerprops = dict(color=[.4, .4, .4], lw=2),
        medianprops = dict(color=salmon, lw=4),
        meanprops = dict(color='r', lw=2),
        manage_ticks=False,
        meanline=True,
        vert = not horizontal,
        )

    for box in boxes["boxes"]:
        box.set(facecolor = "k")

    # kde plot
    kde_kwargs = kde_kwargs or dict(bw=.25)
    plot_kde(
        ax=ax, 
        data=y, 
        vertical=not horizontal, 
        z=x_pos+kde_x, 
        color=color, 
        kde_kwargs=kde_kwargs, 
        zorder=zorder,
        normto=kde_normto,
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

def plot_trial_tracking(ax, trial, tracking_color, start_color, end_color):
    ax.plot(trial.x, trial.y, color=tracking_color)
    ax.scatter(trial.x[0], trial.y[0], color=start_color, zorder=100)
    ax.scatter(trial.x[-1], trial.y[-1], color=end_color, zorder=100)


def plot_threat_tracking_and_angle(dataset, lcolor=tracking_color, rcolor=tracking_color_dark, n_samples=50, **kwargs):
    '''
        Given a dataset it pots the tracking while on T of each trial on the dataset
        alongside the average orientation of the mouse for L vs R trials.
    '''
    axes = generate_figure(ncols=2, figsize=(16, 8))
    trials = dataset.get_orientations_on_T(n_samples=n_samples, **kwargs)

    for i, trial in trials.iterrows():
        if trial.escape_arm == 'right':
            color=rcolor
        elif trial.escape_arm == 'left':
            color= lcolor

        # plot
        axes[0].plot(trial.x, trial.y, color=color)


    L = trials.loc[trials.escape_arm == 'left']
    R = trials.loc[trials.escape_arm == 'right']
    for data, color, lbl in zip((L, R), (lcolor, rcolor), ('left', 'right')):
        angles = np.vstack(data.orientation.values).T

        mu = np.degrees(circmean(np.radians(angles), axis=1))
        sigma = np.degrees(circstd(np.radians(angles), axis=1))
        plot_mean_and_error(mu, sigma, axes[1], color=color, label=lbl)

    axes[0].axis('off')
    axes[1].legend()
    axes[1].set(ylabel='Orientation', xlabel='time', xticks=[0, n_samples], xticklabels=[0, 1])

    return axes