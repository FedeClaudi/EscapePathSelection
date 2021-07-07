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

N_REPS_MODEL = 100 # number of times each model is ran.

# change training settings to reflect parametsr
TRAINING_SETTINGS['episodes'] = 75
TRAINING_SETTINGS['max_n_steps'] = 500

agents =  {
    'QTable':QTableModel,
    'DynaQ_5':DynaQModel,
    'DynaQ_30':DynaQModel,
    'InfluenceZones':InfluenceZones,
    'InfluenceZonesNoSheltVec':InfluenceZones,
}

agent_kwargs = {
    'QTable':dict(),
    'DynaQ_30':dict(n_planning_steps=30),   
    'DynaQ_5':dict(n_planning_steps=5),
    'InfluenceZones':{
        'learning_rate': .8,
        'predict_with_shelter_vector':True
    },
    'InfluenceZonesNoSheltVec':{
        'learning_rate': .8,
        'predict_with_shelter_vector':False
    },
}

# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #
def run():
    for maze_name, maze in zip(('M1', 'M6'), (PsychometricM1, PsychometricM6)):
        logger.info(f'Training on maze: {maze_name} | Number of steps: {TRAINING_SETTINGS["episodes"]} | Max steps per episode {TRAINING_SETTINGS["max_n_steps"]}')

        # results = {
        #     distance_covered=[]
        # }
        for name, model in agents.items():
            logger.info(f'      training agent: {name} | {N_REPS_MODEL} reps')

            # run all instances in parallel
            pool = multiprocessing.Pool(processes = 10)
            arguments = [(maze, model, name, rep) for rep in range(N_REPS_MODEL)]
            run_results = pool.map(run_instance, arguments)

            # get average results for each episode during training
            training_history = [r[0] for r in run_results]
            training_results = {  # average of N episodes for rep of agent (avg across reps)
                'n_steps':[],
                'distance_travelled':[],
                'success':[],
                'n_steps_sem':[],
                'distance_travelled_sem':[],
                'success_sem':[],
            }
            for epn in range(TRAINING_SETTINGS['episodes']):
                training_results['n_steps'].append(
                    np.mean([th.episode_length_history[epn] for th in training_history])
                )
                training_results['distance_travelled'].append(
                    np.mean([th.episode_distance_history[epn] for th in training_history])
                )
                training_results['success'].append(
                    np.mean([th.successes_history[epn] for th in training_history])
                )


                training_results['n_steps_sem'].append(
                    sem([th.episode_length_history[epn] for th in training_history])
                )
                training_results['distance_travelled_sem'].append(
                    sem([th.episode_distance_history[epn] for th in training_history])
                )
                training_results['success_sem'].append(
                    sem([th.successes_history[epn] for th in training_history])
                )
            pd.DataFrame(training_results).to_hdf(f'./cache/{name}_training_on_{maze_name}.h5', key='hdf')

            escape_results = {  # one for each rep of the model
                'rewards': [r[0] for r in run_results],
                'n_steps': [r[2] for r in run_results],
                'escape_arm': [r[3] for r in run_results],
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
            print(f'[dim]Overring default settings value for {param}')
            del settings[param]

        # adjust rewards per model
        if param in rewards.keys():
            print(f'[dim]Overring default reward value for {param}')
            rewards[param] = agent_kwargs[name][param]

    # create an instance
    _maze = maze(rewards)
    _maze.build_graph()
    _maze.shelter_found = False
    agent = model(_maze, name=_maze.name, **settings, **agent_kwargs[name])

    # train
    agent.train(random_start=RANDOM_INIT_POS, episodes=TRAINING_SETTINGS['episodes'])

    # raise ValueError(agent.training_history.episode_length_history)

    # do a play run
    try:
        _, play_steps, play_reward, escape_arm = _maze.play(agent)
    except Exception:
        play_steps = np.nan
        play_reward = np.nan
        escape_arm = np.nan

    print(f'\n[{green}]Finished[/{green}]: Model: [b {pink}]{name}[/b {pink}] - Iteration [{blue_light}]{rep+1}/{N_REPS_MODEL}[/{blue_light}]')
    return (
        agent.training_history,
        play_reward,
        play_steps,
        escape_arm
    )


if __name__ == "__main__":
    run()