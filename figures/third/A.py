# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np

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
    Plot quantized mazes
'''
# %%

logger.remove()
logger.add(sys.stdout, level='INFO')


sessions = (1, 4)
# f, ax = plt.subplots(figsize=(7, 7))
plt.figure(figsize=(9, 9))

maze = PsychometricM1(REWARDS)

plt.imshow(maze.maze)

# model = QTableTracking(
#         PsychometricM1, 
#         'M1',
#         take_all_actions=False,
#         trial_number=4,
#         name=maze.name,
#         **TRAINING_SETTINGS)

# model.plot_tracking(ax=axes[n], plot_raw=False)
# ax.axis('off')
# clean_axes(f)
# plt.show()
# f.savefig(fig_3_path / 'paenl_A_quantized_mazes.svg', format='svg', dpi=dpi)





# %%
plt.figure(figsize=(9, 9))
x, y = np.where(maze.maze == 0)[::-1]
x = x.astype(np.int64)
y = y.astype(np.int64)
plt.scatter(x, y, marker='s', s=250, lw=2, edgecolors='k')
plt.axis('equal')
plt.axis('off')
plt.ylim(50, 0)
plt.savefig(fig_3_path / 'paenl_A_quantized_mazes.eps', format='eps', dpi=dpi)

# %%
