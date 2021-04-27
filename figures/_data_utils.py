import pandas as pd
import numpy as np

from paper import Tracking


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

