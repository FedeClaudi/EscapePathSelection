import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.stats import ttest_ind as ttest
from statsmodels.stats.multitest import multipletests
from brainrender.colors import colorMap
import itertools



def plot_trial_tracking_as_lines(trial, ax, color, N, thick_lw=4, thin_lw=.8, thin_alpha=.6,
                                    outline_color='w', outline_width=2, 
                                    head_size=180,
                                    start_frame = 0,
                                    stop_frame = None, 
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

    if stop_frame is None:
        stop_frame = len(nx)

    # Plot every N thick colored
    for i in np.arange(stop_frame):
        if i < start_frame:
            continue 

        if i % N == 0:
            # Plot outline
            ax.plot([nx[i], bx[i]], [ny[i], by[i]], color=outline_color, lw=thick_lw+outline_width, zorder=1,
                            solid_capstyle='round')
            ax.plot([bx[i], tx[i]], [by[i], ty[i]], color=outline_color, lw=thick_lw+outline_width, zorder=1,
                            solid_capstyle='round')

            # Plot colored line
            if color_by_speed:
                _col = colorMap(speed[i], name=cmap, vmin=0, vmax=speed.max())
                line_col = _col
            else:
                if not isinstance(color, list):
                    line_col = color
                else:
                    line_col = color[i]

            # mark head
            ax.scatter(nx[i], ny[i], color=line_col, lw=outline_width*.75, zorder=3, 
                            s=head_size, edgecolors=outline_color)   

            # Plot body
            ax.plot([nx[i], bx[i]], [ny[i], by[i]], c=line_col, lw=thick_lw, zorder=3,
                            solid_capstyle='round')
            ax.plot([bx[i], tx[i]], [by[i], ty[i]], c=line_col, lw=thick_lw, zorder=3,
                            solid_capstyle='round')

    # Plot all frames
    if isinstance(color, list):
        color = color[0]
    
    if thin_alpha:
        ax.plot([nx[:stop_frame], bx[:stop_frame]], [ny[:stop_frame], by[:stop_frame]], color=color, lw=thin_lw, alpha=thin_alpha, zorder=0,
                        solid_capstyle='round')
        ax.plot([bx[:stop_frame], tx[:stop_frame]], [by[:stop_frame], ty[:stop_frame]], color=color, lw=thin_lw, alpha=thin_alpha, zorder=0,
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

def run_multi_t_test_bonferroni_one_samp_per_item(meandurs):
    """
        it expects a dictionary with 'mazes' as keys and an 
        array of numbers for each maze with some kind measurement. 
    """

    ts, ps, pairs = [], [], []
    for (m1, m2) in itertools.combinations(meandurs.keys(), 2):
        res = ttest(meandurs[m1], meandurs[m2], equal_var=False)
        ts.append(res.statistic)
        ps.append(res.pvalue)
        pairs.append((m1, m2)) 

    significant, pval, _, _ = multipletests(ps, method='bonferroni', alpha=0.05)
    return significant, pval, pairs