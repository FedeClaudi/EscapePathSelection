import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.stats import ttest_ind as ttest
from statsmodels.stats.multitest import multipletests


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