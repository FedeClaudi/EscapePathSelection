from rich import print
from myterial import orange
import matplotlib.pyplot as plt

from loguru import logger
    
import sys
sys.path.append('./')

from fcutils.progress import track

from rl.environment.mazes import PsychometricM1
from rl.environment.render import Render
from experiments._settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from experiments.plot import plot_results, plot_q_and_trial, animate_learning
from models import GraphQModel
'''
    Run an entirely simulted RL experiment: maze are small and made up and it doesn't use
    real tracking data
'''

# get maze
maze = PsychometricM1(REWARDS)
print(f'Starting training of maze [salmon]{maze.name}: [b {orange}]{maze.description}')

# build and draw graph
maze.build_graph()
maze.draw_graph()

# get model
model = GraphQModel(maze, **TRAINING_SETTINGS)
print(f'Running experiment with model\n')
print(model)

# train
logger.info('starting training') 
results = model.train(random_start=RANDOM_INIT_POS, film=False)

# plot
logger.info('starting plotting simulations')
plot_q_and_trial(maze, model)
plt.show()
