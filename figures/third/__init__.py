import sys
from pathlib import Path

if 'darwin' not in sys.platform:
    fig_3_path = Path(r'D:\Dropbox (UCL)\Rotation_vte\Writings\BehavPaper\Figure3')
else:
    fig_3_path = Path('/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Writings/BehavPaper/Figure3')

from rl import environment
from rl.environment.environment import Status
from rl.environment.mazes import PsychometricM1, PsychometricM6
from rl.models import QTableModel, DynaQModel, InfluenceZones, QTableTracking, DynaQTracking, InfluenceZonesTracking

from myterial import salmon, indigo, green


MODELS = ['QTable', 'DynaQ_20', 'InfluenceZonesNoSheltVec',]
MAZES = ['M1', 'M6']
MODELS_COLORS = (salmon, indigo, green)



accepted_sessions = dict(
    M1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    M6 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 31, 33],
)




