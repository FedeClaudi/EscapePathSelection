# %%
import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

from fcutils.plot.figure import clean_axes

from figures.third import PsychometricM1, PsychometricM6, QTableTracking, fig_3_path
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from figures.settings import dpi

''' 
number of state changes

M1: 753.49 +/- 264.938
M6: 790.60 +/- 269.019
'''

logger.remove()
logger.add(sys.stdout, level='INFO')

f, axes = plt.subplots(ncols=2, figsize=(16, 9), sharex=True, sharey=True)
for n, (_maze, fname) in enumerate(zip((PsychometricM6, PsychometricM1), ('M6', 'M1'))):

    maze = _maze(REWARDS)
    n_state_changes = []
    n_sessions, n_accepted = 0, 0
    accepted_sessions = []
    for session_n in range(100):

        try:
            model = QTableTracking(
                    maze, 
                    fname,
                    take_all_actions=False,
                    trial_number=session_n,
                    name=maze.name,
                    **TRAINING_SETTINGS)

        except Exception: 
            break
        else:
            # model.plot_tracking()
            # plt.show()
            n_sessions += 1

        logger.info(f'|Maze {maze.name}| session {session_n} - {len(model.tracking)} state transitions | {len(model.tracking_errors)} errors')
        if len(model.tracking_errors) < 5:
            n_accepted += 1
            accepted_sessions.append(session_n)
            n_state_changes.append(len(model.tracking))

    axes[n].hist(n_state_changes, bins=10)
    axes[n].axvline(np.mean(n_state_changes), color='r')
    axes[n].set(title=maze.name, ylabel='counts', xlabel='# state changes')
    logger.info(f'-- maze: {maze.name} -- {n_accepted}/{n_sessions} accepted')
    logger.info(f'-- maze: {maze.name} -- mean number of steps: {np.mean(n_state_changes):.2f} +/- {np.std(n_state_changes):.3f}\n\n')
    print(f'{maze.name} - accepted sessions {accepted_sessions}')


clean_axes(f)
plt.show()
f.savefig('/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Writings/BehavPaper/Revision/Plots/panel_C_guided_exploration_n_state_transitions_histograms.eps', format='eps', dpi=dpi)



# %%
