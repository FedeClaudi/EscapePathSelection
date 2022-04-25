import sys
sys.path.append("./")
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


from paper import Session, Stimuli, TrackingData, Tracking, Trials
from ..figures import DataSet

"""
Code to organize the data in a way where they can be shared upon publication.

"""


fld = Path(r"D:\Dropbox (UCL)\Rotation_vte\Writings\BehavPaper\data_share")


logger.debug('Loading M1')
M1 = DataSet('M1', Trials.get_by_condition(maze_design=1, escape_duration=10))
logger.debug('Loading M2')
M2 = DataSet('M2', Trials.get_by_condition(maze_design=2, escape_duration=10))
logger.debug('Loading M3')
M3 = DataSet('M3', Trials.get_by_condition(maze_design=3, escape_duration=10))
logger.debug('Loading M4')
M4 = DataSet('M4', Trials.get_by_condition(maze_design=4, escape_duration=10))
logger.debug('Loading M6')
M6 = DataSet('M6', Trials.get_by_condition(maze_design=6, escape_duration=10))


# %%
# get all the session names and on which arena they are from trials


# %%
# prepare and save metadata file



# %%
# save tracking data


# %%
# save stimuli