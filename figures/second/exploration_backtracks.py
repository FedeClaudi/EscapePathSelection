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

def get_at_roi(rois, target):
    """
        Timepoints of when the mouse is in a given ROI
    """
    in_target = np.zeros(len(rois))
    in_target[rois == target] = 1
    at_target = np.where(np.diff(in_target)>0)[0]

    return at_target



# %%
complete, incomplete = [], []

for n, data in enumerate(datasets):
    print(n)

    for sess in data.sessions:
        try:
            exploration = pd.Series((Explorations & f'session_name="{sess}"').fetch1())
        except Exception:
            print(f'No exploration for {sess}')
            continue

        # get all the times the mouse reaches the shelter and the threat platform
        at_shelter = get_at_roi(exploration.maze_roi, 0)
        at_threat = get_at_roi(exploration.maze_roi, 1)
        at_roi = np.hstack([at_shelter, at_threat])
        at_roi_identity = np.hstack([np.zeros_like(at_shelter), np.ones_like(at_threat)])

        at_roi_identity = at_roi_identity[np.argsort(at_roi)]
        at_roi = np.sort(at_roi)

        # loop over each time the mouse enters a target and see what happens next
        complete_sess, incomplete_sess = 0, 0
        for i, (roi, time) in enumerate(zip(at_roi_identity[:-1], at_roi[:-1])):
            next_roi = at_roi_identity[i+1]
            next_time = at_roi[i+1]

            # check for errors
            if i > 0:
                if abs(time - at_roi[i-1]) < 30:
                    continue

            if abs(next_time - time) < 50:
                # too brief
                continue

            # check if there was an error
            if exploration.body_tracking[time, 0] < 200 or exploration.body_tracking[time, 0] > 800:
                # print('s4')
                continue
            if exploration.body_tracking[next_time, 0] < 200 or exploration.body_tracking[next_time, 0] > 800:
                # print('s4')
                continue

            # check if mouse returned to same ROI
            if next_roi == roi:
                incomplete_sess += 1
            else:
                complete_sess += 1
        complete.append(complete_sess)
        incomplete.append(incomplete_sess)

            

        # break
    # break
print(np.mean(complete), np.mean(incomplete))
print(np.std(complete), np.std(incomplete))

#%%

plt.hist(complete, alpha=.5)
plt.hist(incomplete, alpha=.5)
plt.show()