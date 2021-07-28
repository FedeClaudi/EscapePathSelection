
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


from fcutils.plot.figure import clean_axes
from fcutils.plot.elements import plot_mean_and_error
from fcutils.maths import rolling_mean

import sys
from pathlib import Path
import os
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')


from myterial.utils import map_color


from environment.actions import Actions

from figures.third import MODELS_COLORS, MODELS, MAZES, fig_3_path
from figures.settings import dpi
from figures.third import PsychometricM1, PsychometricM6, QTableModel, DynaQModel, InfluenceZones, Status, QTableTracking, DynaQTracking, InfluenceZonesTracking
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from figures.settings import dpi


'''
    Plot the policy maps of trained agents
'''
# %%

logger.remove()
logger.add(sys.stdout, level='INFO')
# -------------------------------- parameters -------------------------------- #

# change training settings to reflect parametsr
TRAINING_SETTINGS['episodes'] = 250
TRAINING_SETTINGS['max_n_steps'] = 500


def draw_arrows(agent, ax):
    actions = Actions()
    for cell in agent.environment.empty:
        q = agent.q(cell)

        if len(q) == 0:
            continue

        a = np.nonzero(q == np.max(q))[0]

        qmin, qmax = np.min(q), np.max(q)

        for action in np.arange(len(q)):
            if q[action] != qmax:
                continue
            dx, dy = actions[action].shift / 6

            angle = np.degrees(np.arctan2(*actions[action].shift))
            color = map_color(angle, name='bwr', vmin=-180, vmax=180)
            ax.scatter(*cell, color=color, s=250, marker='s', lw=1, edgecolors='k')
    
    # plot a legent
    for action in actions:
        dx, dy = actions[action].shift * 2
        angle = np.degrees(np.arctan2(*actions[action].shift))
        color = map_color(angle, name='bwr', vmin=-180, vmax=180)

        ax.arrow(24, 23, dx, dy, color=color, head_width=1, head_length=1, width=.3, zorder=2, lw=1, ec='k')
    ax.scatter(24, 23, color='w', zorder=100, s=400, lw=1, ec='k')


def plot(agent, name, exploration):
    f, ax = plt.subplots(figsize=(9, 9))
    
    draw_arrows(agent, ax)

    ax.set(ylim=[50, 0], title=name)
    ax.axis('equal')    
    ax.axis('off')



    # draw maze
    x, y = np.where(agent.environment.maze == 0)[::-1]
    ax.scatter(
        x,
        y,
        color=[.8, .8, .8],
        lw=1, edgecolors=['k'], marker='s', s=250, zorder=-1
    )
    ax.set(ylim=[50, 0], title=name)
    ax.axis('equal')
    ax.axis('off')

    f.savefig(fig_3_path / f'{name}_{exploration}_Q_map.eps', format='eps', dpi=dpi)




# %%
# ---------------------------------------------------------------------------- #
#                                   FREE EXPL                                  #
# ---------------------------------------------------------------------------- #

agents =  {
    'QTable':QTableModel,
    'DynaQ_20': DynaQModel,
    'InfluenceZonesNoSheltVec':InfluenceZones,
}

agent_kwargs = {
    'QTable':dict(learning_rate=.9, penalty_move = 1e-8),
    'DynaQ_20':dict(n_planning_steps=20),   
    'InfluenceZonesNoSheltVec':dict(predict_with_shelter_vector=False, learning_rate=.2, discount=.8),
}
maze = PsychometricM1

for n, (name, model) in enumerate(agents.items()):

    logger.info(f'      training agent: {name} ')

    # remove duplicate parameters
    settings = TRAINING_SETTINGS.copy()
    rewards = REWARDS.copy()
    for param in agent_kwargs[name].keys():
        if param in settings.keys():
            # print(f'[dim]Overring default settings value for {param}')
            del settings[param]

        # adjust rewards per model
        if param in rewards.keys():
            # print(f'[dim]Overring default reward value for {param}')
            rewards[param] = agent_kwargs[name][param]

    # create an instance
    status = Status.LOSE
    while status != Status.WIN:  # keep going until we found a model that won
        _maze = maze(rewards)
        
        _maze.build_graph()
        _maze.shelter_found = False
        agent = model(_maze, name=_maze.name, **settings, **agent_kwargs[name])

        # train
        agent.train(random_start=RANDOM_INIT_POS, episodes=TRAINING_SETTINGS['episodes'], test_performance=True)
        status, play_steps, play_reward, escape_arm, states = _maze.play(agent, start_cell=_maze.START)

    # plot
    plot(agent, name, 'free')



# %%
# ---------------------------------------------------------------------------- #
#                                 GUIDED EXPL                                  #
# ---------------------------------------------------------------------------- #

sessions = [24]

agents =  {
    'QTable':QTableTracking,
    'DynaQ_20':DynaQTracking,
    'InfluenceZonesNoSheltVec':InfluenceZonesTracking,
}

agent_kwargs = {
    'QTable':dict(learning_rate=.9),
    'DynaQ_20':dict(n_planning_steps=20),   
    'InfluenceZonesNoSheltVec':dict(predict_with_shelter_vector=False, learning_rate=.2, discount=.8),
}

# iterate over mazes and models

for name, model in agents.items():
    # agent specific settings
    agent_settings = TRAINING_SETTINGS.copy()
    agent_rewards = REWARDS.copy()
    for param in agent_kwargs[name].keys():
        if param in agent_settings.keys():
            del agent_settings[param]

        # adjust rewards per model
        if param in agent_rewards.keys():
            agent_rewards[param] = agent_kwargs[name][param]

    # iterate over trials
    trajectories = []
    for session_number in sessions:
        # instantiate model and maze
        _maze = maze(agent_rewards)
        _model = model(
                    _maze, 
                    'M1',
                    take_all_actions=False,
                    trial_number=session_number,
                    name=_maze.name,
                    **agent_settings, **agent_kwargs[name])

        # train
        _model.train(film=False)

        # test
        status, play_steps, play_reward, escape_arm, states = _maze.play(_model, start_cell=_maze.START)
        trajectories.append(states)

    plot(_model, name, 'guided')




# %%
