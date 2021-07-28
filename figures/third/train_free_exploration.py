import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
from rich import print
from loguru import logger
import sys 
import multiprocessing
import scipy
import copy
import pandas as pd
from scipy.stats import sem


from myterial import salmon, pink, blue_light, green
from fcutils.maths import rolling_mean

from figures.third import PsychometricM1, PsychometricM6, QTableModel, DynaQModel, InfluenceZones
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS


'''
    Train the three agents on each of the two mazes 
'''

logger.remove()
logger.add(sys.stdout, level='INFO')
# -------------------------------- parameters -------------------------------- #

N_REPS_MODEL = 64 # number of times each model is ran.

# change training settings to reflect parametsr
TRAINING_SETTINGS['episodes'] = 250
TRAINING_SETTINGS['max_n_steps'] = 500

agents =  {
    # 'QTable':QTableModel,
    # 'DynaQ_20':DynaQModel,
    'InfluenceZonesNoSheltVec':InfluenceZones,
}

agent_kwargs = {
    'QTable':dict(learning_rate=.9, penalty_move = 1e-8),
    'DynaQ_20':dict(n_planning_steps=20),   
    'InfluenceZonesNoSheltVec':dict(predict_with_shelter_vector=False, learning_rate=.2, discount=.8),
}

# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #
def run():
    for maze_name, maze in zip(('M1', 'M6'), (PsychometricM1, PsychometricM6)):
        if maze_name == 'M1':
            continue

        logger.info(f'Training on maze: {maze_name} | Number of steps: {TRAINING_SETTINGS["episodes"]} | Max steps per episode {TRAINING_SETTINGS["max_n_steps"]}')

        for name, model in agents.items():
            logger.info(f'      training agent: {name} | {N_REPS_MODEL} reps | on {maze_name}')

            # run all instances in parallel
            pool = multiprocessing.Pool(processes = 8)
            arguments = [(maze, model, name, rep) for rep in range(N_REPS_MODEL)]
            run_results = pool.map(run_instance, arguments)

            # get average results for each episode during training
            training_history = [r[0] for r in run_results]
            training_results = {  # average of N episodes for rep of agent (avg across reps for each episode)
                'n_steps':[],
                'distance_travelled':[],
                'success':[],
                'n_steps_sem':[],
                'distance_travelled_sem':[],
                'success_sem':[],
                'play_status':[],
                'play_status_sem':[],
                'play_steps':[],
                'play_steps_sem':[],
            }
            for epn in range(TRAINING_SETTINGS['episodes']):
                keys = [
                    ('n_steps', 'episode_length_history'),
                    ('distance_travelled', 'episode_distance_history'),
                    ('success', 'successes_history'),
                    ('play_status', 'play_status_history'),
                    ('play_steps', 'play_steps_history'),
                ]
                for k, v in keys:
                    training_results[k].append(np.mean([th.data[v][epn] for th in training_history]))
                    training_results[k+'_sem'].append(sem([th.data[v][epn] for th in training_history]))

            pd.DataFrame(training_results).to_hdf(f'./cache/{name}_training_on_{maze_name}.h5', key='hdf')

            escape_results = {  # one for each rep of the model
                'rewards': [r[1] for r in run_results],
                'n_steps': [r[2] for r in run_results],
                'escape_arm': [r[3] for r in run_results],
                'status': [r[4] for r in run_results],
            }
            pd.DataFrame(escape_results).to_hdf(f'./cache/{name}_escape_on_{maze_name}.h5', key='hdf')


# ------------------------------- run instance ------------------------------- #
def run_instance(args):
    '''
        Runs a single instance and returns the results, used for
        running multiple instances in parallel.
    '''
    maze, model, name, rep = args
    print(f'\n[{salmon}]Starting[/{salmon}]: Model: [b {pink}]{name}[/b {pink}] - Iteration [{blue_light}]{rep+1}/{N_REPS_MODEL}[/{blue_light}]')

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

    # raise ValueError(agent.training_history.episode_length_history)

    # do a play run
    try:
        status, play_steps, play_reward, escape_arm, states = _maze.play(agent, start_cell=_maze.START)
    except Exception as e:
        logger.warning(f'exception when running the model in play mode:\n{e}')
        play_steps = np.nan
        play_reward = np.nan
        escape_arm = np.nan
        status = np.nan

    print(f'\n[{green}]Finished[/{green}]: Model: [b {pink}]{name}[/b {pink}] - Iteration [{blue_light}]{rep+1}/{N_REPS_MODEL}[/{blue_light}] - play status: {status}')
    return (
        agent.training_history,
        play_reward,
        play_steps,
        escape_arm,
        status,
    )


if __name__ == "__main__":
    run()