# %%
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.path import from_yaml
from fcutils.plot.figure import clean_axes

from paper import Explorations

from figures.first import M1, M2, M3, M4,M6
from figures._plot_utils import triple_plot
from figures.settings import dpi
from figures.bayes import Bayes
from figures.glm import GLM
from figures.second import fig_2_path

datasets = (M1, M2, M3, M4, M6)


"""  
    Look at frequency of times the mice backtrack during exzploratio
    (as opposed to running a whole path)

"""
# %%
def get_trips(rois, source, target, tracking):
    '''
        Given a list of roi indices for where the mouse is at for every frame, 
        it gets all the times that the mouse did a trip source -> target
    '''
    in_target = np.zeros(len(rois))
    in_target[rois == target] = 1

    at_target = np.where(np.diff(in_target)>0)[0]

    trips = dict(complete=[], incomplete=[])
    for tgt in at_target:
        # go back in time and get the last time the animal was at source
        try:
            at_source = np.where(rois[:tgt] == source)[0][-1]
        except  IndexError:
            # print('s1')
            continue

        # if the mouse gets to tgt any other time during the trip, ignore
        if np.any(rois[at_source:tgt-3] == target):
            # print('s2')
            continue

        if tgt - at_source < 50: 
            # print('s3')
            # too fast
            continue

        # get arm
        if np.min(tracking[at_source: tgt, 0]) <= 400:
            arm = 'left'
        else:
            arm = 'right'

        # check if there was an error
        if tracking[tgt, 0] < 200 or tracking[tgt, 0] > 800:
            # print('s4')
            continue

        # get distance travelled
        # x = tracking[at_source:tgt, 0]
        # y = tracking[at_source:tgt, 1]
        # dist = get_dist(x, y)
        dist = np.sum(tracking[at_source:tgt, 2]) * 0.22
        if dist < 25:
            # print('s5')
            continue  # too short

        trips.append((at_source, tgt, arm, dist))
    return trips