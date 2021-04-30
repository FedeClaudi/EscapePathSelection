from figures import MAIN_FLD
from loguru import logger

fig_1_path = MAIN_FLD / 'figure1'
fig_1_plots = fig_1_path / 'plots'


import sys
sys.path.append('./')

# import psychometric trials
from paper import Trials
from figures.settings import max_escape_duration
from figures.dataset import DataSet

logger.debug('Loading M1')
M1 = DataSet('M1', Trials.get_by_condition(maze_design=1, escape_duration=max_escape_duration, clean=True))
M1.clean()

logger.debug('Loading M2')
M2 = DataSet('M2', Trials.get_by_condition(maze_design=2, escape_duration=max_escape_duration, clean=True))
M2.clean()

logger.debug('Loading M3')
M3 = DataSet('M3', Trials.get_by_condition(maze_design=3, escape_duration=max_escape_duration, clean=True))
M3.clean()

logger.debug('Loading M4')
M4 = DataSet('M4', Trials.get_by_condition(maze_design=4, escape_duration=max_escape_duration, clean=True))
M4.clean()

logger.debug('Loading M6')
M6 = DataSet('M6', Trials.get_by_condition(maze_design=6, escape_duration=max_escape_duration, clean=True))
M6.clean()
