import sys
sys.path.append('./')
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import pandas as pd

from fcutils.plot.figure import clean_axes

from figures.third import PsychometricM1, PsychometricM2, PsychometricM3, QTableTracking, DynaQTracking, InfluenceZonesTracking, fig_3_path, accepted_sessions, MAZES, Status
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from figures.settings import dpi



maze = PsychometricM3(None)

good_sessions = []
for trial in range(100):
    try:
        agent = QTableTracking(maze, 'M3', trial_number=trial)
    except Exception:
        continue

    print(f'Number of tracking errors: {len(agent.tracking_errors)}')
    # agent.plot_tracking()

    # plt.show()

    if len(agent.tracking_errors) < 2:
        good_sessions.append(trial)

print(good_sessions)




