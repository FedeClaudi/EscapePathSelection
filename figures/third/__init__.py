import sys

# import RL code from https://github.com/FedeClaudi/Reinforcement-Learning-Maze.git
sys.path.append(r'C:\Users\Federico\Documents\GitHub\mazeRL')

from environment.mazes import PsychometricM1, PsychometricM6
from models import QTableModel, DynaQModel, InfluenceZones
