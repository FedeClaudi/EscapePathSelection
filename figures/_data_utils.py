import pandas as pd
import numpy as np

from paper import Tracking


def register_in_time(trials, n_samples):
    '''
        Given a list of 1d numpy arrays of different length,
        this function returns an array of shape (n_samples, n_trials) so
        that each trial has the same number of samples and can thus be averaged
        nicely
    '''
    target = np.zeros((n_samples, len(trials)))
    for trial_n, trial in enumerate(trials):
        n = len(trial)
        for i in range(n_samples):
            idx = int(np.floor(n * (i / n_samples)))
            target[i, trial_n] = trial[idx]
    return target

def get_recording_body_tracking(recording):
    '''Returns the XYS tracking of the mouse body for a recording'''

    data = Tracking * Tracking.BodyPart & f'bpname="body"' & f'recording_uid="{recording}"'

    tracking = pd.DataFrame(data.fetch())

    try:
        return np.vstack(*tracking[['x', 'y', 'speed']].values).T
    except  TypeError:
        return None


def get_recording_maze_component(recording):
    '''
        Given a recording it returns the maze component the mouse
        is on at any given frame.
    '''
    data = Tracking * Tracking.BodyPart & f'bpname="body"' & f'recording_uid="{recording}"'

    tracking = pd.DataFrame(data.fetch())

    try:
        return tracking['roi'][0]
    except  TypeError:
        return None


def get_pR_from_trials_df(trials):
    ''' 
        Given a dataframe of trials data it computes p(R)
    '''
    R = len(trials.loc[trials.escape_arm == 'right'])
    L = len(trials.loc[trials.escape_arm == 'left'])
    return R / (R+L)

def get_number_per_arm_fro_trials_df(trials):
    R = len(trials.loc[trials.escape_arm == 'right'])
    L = len(trials.loc[trials.escape_arm == 'left'])

    return L, R