import sys
from pathlib import Path
# import RL code from https://github.com/FedeClaudi/Reinforcement-Learning-Maze.git
if 'darwin' not in sys.platform:
    sys.path.append(r'C:\Users\Federico\Documents\GitHub\mazeRL')
    fig_3_path = Path(r'D:\Dropbox (UCL)\Rotation_vte\Writings\BehavPaper\Figure3')
else:
    sys.path.append('/Users/federicoclaudi/Documents/Github/Reinforcement-Learning-Maze')
    fig_3_path = Path('/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Writings/BehavPaper/Figure3')

from environment.mazes import PsychometricM1, PsychometricM6
from models import QTableModel, DynaQModel, InfluenceZones, QTableTracking, DynaQTracking, InfluenceZonesTracking

from myterial import salmon, indigo, green_dark, green, indigo_dark


MODELS = ['QTable', 'DynaQ_5', 'DynaQ_30', 'InfluenceZones', 'InfluenceZonesNoSheltVec']
MAZES = ['M1', 'M6']
MODELS_COLORS = (salmon, indigo, indigo_dark, green, green_dark)


