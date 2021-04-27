from tqdm import tqdm
import pyinspect as pi
import numpy as np
import matplotlib.pyplot as plt
from rich import print

from brainrender.colors import makePalette, colors
import fcutils
from fcutils.plotting.utils import clean_axes

from pyinspect._colors import *
pi.install_traceback()

from PIL import ImageColor



def rgb(col):
    return ImageColor.getrgb(col.lower())


# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #

params = dict(
    dur_short = 5, # duration of escape on short arm (s)
    dur_long = 7, # duration of escape on long arm (s)

    rt_range = np.linspace(0, 3.5, 100),  # range of possible RT

    curve_type='linear',  # type of accuracy curve
    lin_slope = np.linspace(10, 75, 300),  # slope of linear rt/acc curve
    
    alphas = np.linspace(0.1, 1, 300),  # for plotting
)


# ---------------------------------------------------------------------------- #  
#                                    GETTERS                                   #
# ---------------------------------------------------------------------------- #

def get_rt_acc_curve(*args, **kwargs):
    def linear(slope=None):
        acc =  params['rt_range'] * slope + 50  # at worse they sould do 50-50
        acc[acc > 100] = 100
        acc[acc < 0] = 0
        return acc

    def log():
        raise NotImplementedError

    def sigmoid():
        raise Exception

    if params["curve_type"] == 'linear':
        acc =  linear(**kwargs)
    elif params["curve_type"] == 'log':
        acc =  log(**kwargs)
    elif params["curve_type"] == 'sigmoid':
        acc =  sigmoid(**kwargs)
    else:
        raise ValueError(f'What curve, sorry? {params["curve_type"]}')

    rt_acc_lookup = {rt:a/100 for rt,a in zip(params['rt_range'], acc)}

    return acc, params['rt_range'][np.argmax(acc)], rt_acc_lookup

def get_duration_rt_curve(acc):
    """  
        For a given RT, the expected escape duration is:

        accuracy * dur_short + ( 1 - accuracy) * dur_long + RT
    """
    acc = acc / 100  # it's normally expressed in percentage
    dur = acc * params['dur_short'] + (1 - acc) * params['dur_long'] + params['rt_range']

    return dur, params['rt_range'][np.argmin(dur)]

# ---------------------------------------------------------------------------- #
#                                   PLOTTERS                                   #
# ---------------------------------------------------------------------------- #
def plot_rt_acc_curve(acc, ax):
    ax.plot(params['rt_range'], acc, lw=2, color=_colors[n], alpha=.5)
    # ax.axvline(acc_max, lw=1, color=[.6, .6, .6])
    ax.set(title='Reaction time accuracy curve', xlabel='RT (s)', ylabel='Accuracy (%)')

def plot_rt__dur_curve(dur, ax):
    ax.plot(params['rt_range'], dur, lw=2, color=_colors[n], alpha=.5)
    # ax.axvline(min_dur, lw=1, color=[.6, .6, .6])
    ax.set(title='Reaction time escape duration cuve', xlabel='RT (s)', ylabel='Esc. duration (s)')

def make_fig():
    f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(16, 9))
    axarr = axarr.flatten()

    axarr[1].axhline(params['dur_short'], lw=1, color=[.6, .6, .6])
    axarr[1].axhline(params['dur_long'], lw=1, color=[.6, .6, .6])

    axarr[2].set(title='Best RT vs Accuracy=100 RT', ylabel='Min dur. RT', xlabel='Max accuracy RT')
    return f, axarr

def plot_all(axarr, acc, dur):


    plot_rt_acc_curve(acc, axarr[0])
    plot_rt__dur_curve(dur, axarr[1])



    axarr[2].scatter(acc_max, min_dur, s=100, color=_colors[n], alpha=.5)

    return f

# ---------------------------------------------------------------------------- #
#                                    RUNNING                                   #
# ---------------------------------------------------------------------------- #

f, axarr = make_fig()

N = len(params['lin_slope'])
_colors = makePalette(N, rgb(salmon), rgb(lightgreen), rgb(lilla))

for n, slope in tqdm(enumerate(params['lin_slope'])):
    acc, acc_max, rt_acc_lookup = get_rt_acc_curve(slope=slope)
    dur, min_dur = get_duration_rt_curve(acc)

    plot_all(axarr, acc, dur)

clean_axes(f)
f.tight_layout()
plt.show()
