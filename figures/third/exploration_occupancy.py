
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



from figures.third import MODELS_COLORS, MODELS, MAZES, fig_3_path
from figures.settings import dpi
from figures.third import PsychometricM1, PsychometricM6, QTableModel, DynaQModel, InfluenceZones, Status, QTableTracking, DynaQTracking, InfluenceZonesTracking
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from figures.settings import dpi


'''
    Plot the maze occupancy during trainig for all models and both free and guided exploration
'''
# %%

logger.remove()
logger.add(sys.stdout, level='INFO')
# -------------------------------- parameters -------------------------------- #

# change training settings to reflect parametsr
TRAINING_SETTINGS['episodes'] = 250
TRAINING_SETTINGS['max_n_steps'] = 500




def plot_heatmap(states_counts, name, exploration):
    norm=mpl.colors.LogNorm(vmin=0, vmax=500)

    f, ax = plt.subplots()
    ax.scatter(
        [k[0] for k,v in states_counts.items() if v>0],
        [k[1] for k,v in states_counts.items() if v>0], 
        c=[v for v in states_counts.values() if v>0],
        vmin=1, vmax=500, cmap='bwr', lw=1, edgecolors=['k'], marker='s', s=65, norm=norm,
    )
    ax.set(ylim=[50, 0], title=name + '  ' + exploration)
    ax.axis('equal')
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cmap = mpl.cm.bwr
    # norm = mpl.colors.Normalize(vmin=1, vmax=500)

    f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax, orientation='vertical', label='# visits')
    f.savefig(fig_3_path / f'{name}_{exploration}_exploration_occupancy.eps', format='eps', dpi=dpi)


# %%
# ---------------------------------------------------------------------------- #
#                                   FREE EXPL                                  #
# ---------------------------------------------------------------------------- #

agents =  {
    # 'QTable':QTableModel,
    # 'DynaQ_20': DynaQModel,
    'InfluenceZonesNoSheltVec':InfluenceZones,
}

agent_kwargs = {
    'QTable':dict(learning_rate=.9, penalty_move = 1e-8),
    'DynaQ_20':dict(n_planning_steps=20),   
    'InfluenceZonesNoSheltVec':dict(predict_with_shelter_vector=False, learning_rate=.2, discount=.8),
}
maze = PsychometricM1

for n, (name, model) in enumerate(agents.items()):
    status = Status.LOSE
    while status != Status.WIN:  # keep going until we found a model that won
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
        _maze = maze(rewards)
        _maze.build_graph()
        _maze.shelter_found = False
        agent = model(_maze, name=_maze.name, **settings, **agent_kwargs[name])

        # train
        agent.train(random_start=RANDOM_INIT_POS, episodes=TRAINING_SETTINGS['episodes'], test_performance=True)

        # test
        status, play_steps, play_reward, escape_arm, states = _maze.play(agent, start_cell=_maze.START)


    # plot heatmap of state history
    visited = [tuple(state) for state in agent.training_history.state_history]
    states_counts = {cell:visited.count(cell) for cell in agent.environment.cells}
    plot_heatmap(states_counts, name, 'free')


# %%
# ---------------------------------------------------------------------------- #
#                                 GUIDED EXPL                                  #
# ---------------------------------------------------------------------------- #


session_number = 24

# instantiate model and maze
_maze = maze(agent_rewards)
_model = QTableTracking(
            _maze, 
            'M1',
            take_all_actions=False,
            trial_number=session_number,
            name=_maze.name,
            **agent_settings, **agent_kwargs[name])

# plot heatmap of state history
visited = [tuple(state) for state in _model.tracking]
states_counts = {cell:visited.count(cell) for cell in _model.environment.cells}
plot_heatmap(states_counts, '', 'guided_exploration')






# %%
