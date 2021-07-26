
import sys
sys.path.append('./')
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

from fcutils.plot.figure import clean_axes

from figures.third import PsychometricM1, PsychometricM6, QTableTracking, fig_3_path
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from figures.settings import dpi

'''
    Plot quantized mazes
'''

logger.remove()
logger.add(sys.stdout, level='INFO')


sessions = (1, 4)
f, axes = plt.subplots(ncols=2, figsize=(16, 9), sharex=True, sharey=True)
for n, (_maze, fname) in enumerate(zip((PsychometricM6, PsychometricM1), ('M6', 'M1'))):

    maze = _maze(REWARDS)


    model = QTableTracking(
            maze, 
            fname,
            take_all_actions=False,
            trial_number=sessions[n],
            name=maze.name,
            **TRAINING_SETTINGS)

    model.plot_tracking(ax=axes[n], plot_raw=False)
    axes[n].axis('off')

clean_axes(f)
plt.show()
f.savefig(fig_3_path / 'paenl_A_quantized_mazes.svg', format='svg', dpi=dpi)



