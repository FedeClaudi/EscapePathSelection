import sys
from pathlib import Path
# import RL code from https://github.com/FedeClaudi/Reinforcement-Learning-Maze.git
if 'darwin' not in sys.platform:
    sys.path.append(r'C:\Users\Federico\Documents\GitHub\mazeRL')
    fig_3_path = Path(r'D:\Dropbox (UCL)\Rotation_vte\Writings\BehavPaper\Figure3')
else:
    sys.path.append('/Users/federicoclaudi/Documents/Github/Reinforcement-Learning-Maze')
    fig_3_path = Path('/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Writings/BehavPaper/Figure3')

from environment.environment import Status
from environment.mazes import PsychometricM1, PsychometricM6
from models import QTableModel, DynaQModel, InfluenceZones, QTableTracking, DynaQTracking, InfluenceZonesTracking

from myterial import salmon, indigo, green


MODELS = ['QTable', 'DynaQ_20', 'InfluenceZonesNoSheltVec',]
MAZES = ['M1', 'M6']
MODELS_COLORS = (salmon, indigo, green)



accepted_sessions = dict(
    M1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    M6 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 31, 33],
)




