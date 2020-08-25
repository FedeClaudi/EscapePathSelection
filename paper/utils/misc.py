import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.stats import ttest_ind as ttest
from statsmodels.stats.multitest import multipletests
from brainrender.colors import colorMap

def plot_trial_tracking_as_lines(trial, ax, color, N, thick_lw=4, thin_lw=.8, thin_alpha=.6,
                                    outline_color='w', outline_width=2, 
                                    color_by_speed=False, cmap=None):
    """
        Given a Trial's data it plots the tracking as
        line showing the body-head and tial-body axes. 
        Most lines are small and fain, but every N line
        a thicker and opaque one is shown.
        Looks really nice

        if color_by_speed is True, the color of the thick lines
        depends on the speed using  colorMap(cmap)
    """
    nx, ny = trial.neck_xy[:, 0], trial.neck_xy[:, 1]
    bx, by = trial.body_xy[:, 0], trial.body_xy[:, 1]
    tx, ty = trial.tail_xy[:, 0], trial.tail_xy[:, 1]
    speed = trial.body_speed

    # Plot every N thick colored
    for i in np.arange(len(nx)):
        if i % N == 0:
            # Plot outline
            ax.plot([nx[i], bx[i]], [ny[i], by[i]], color=outline_color, lw=thick_lw+outline_width, zorder=1,
                            solid_capstyle='round')
            ax.plot([bx[i], tx[i]], [by[i], ty[i]], color=outline_color, lw=thick_lw+outline_width, zorder=1,
                            solid_capstyle='round')

            # Plot colored line
            if color_by_speed:
                _col = colorMap(speed[i], name=cmap, vmin=0, vmax=speed.max())
                ax.plot([nx[i], bx[i]], [ny[i], by[i]], c=_col, lw=thick_lw, zorder=3,
                                solid_capstyle='round')
                ax.plot([bx[i], tx[i]], [by[i], ty[i]], c=_col, lw=thick_lw, zorder=2,
                                solid_capstyle='round')
            else:
                ax.plot([nx[i], bx[i]], [ny[i], by[i]], c=color, lw=thick_lw, zorder=3,
                                solid_capstyle='round')
                ax.plot([bx[i], tx[i]], [by[i], ty[i]], c=color, lw=thick_lw, zorder=2,
                                solid_capstyle='round')

    # Plot all frames
    ax.plot([nx, bx], [ny, by], color=color, lw=thin_lw, alpha=thin_alpha, zorder=0,
                    solid_capstyle='round')
    ax.plot([bx, tx], [by, ty], color=color, lw=thin_lw, alpha=thin_alpha, zorder=0,
                    solid_capstyle='round')

def resample_list_of_arrayes_to_avg_len(lst, N=None, interpolate=False):
    """
        Given a list of arrays of varying length, this function
        resamples them so that they all have the 
        average length.
        Then it returns the vstack of the array
    """
    if N is None:
        N = np.mean([len(x) for x in lst]).astype(np.int32)

    if interpolate:
        lst = [pd.Series(x).interpolate() for x in lst]

    return np.vstack([resample(X, N) for X in lst])


def run_multi_t_test_bonferroni(meandurs):
    """
        It expects a dictionary with 'mazes' as keys, for each maze
        another dictionary with 'l' and 'r' as keys with the quantity of interest
        for left vs right paths.


        Paired t-test with bonferroni correction
         to see if difference in duration betwee left and right paths is significant. 

        https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f
    """
    ts, ps = [], []

    for maze in meandurs.keys():
        res = ttest(meandurs[maze]['l'], meandurs[maze]['r'], equal_var =False)
        ts.append(res.statistic)
        ps.append(res.pvalue)

    significant, pval, _, _ = multipletests(ps, method='bonferroni', alpha=0.05)
    return significant, pval