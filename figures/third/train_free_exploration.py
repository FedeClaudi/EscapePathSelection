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

from figures.third import PsychometricM1, PsychometricM2, PsychometricM3, QTableModel, DynaQModel, InfluenceZones
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS


'''
    Train the three agents on each of the two mazes 
'''

logger.remove()
logger.add(sys.stdout, level='INFO')
# -------------------------------- parameters -------------------------------- #

# N_REPS_MODEL = 64 # number of times each model is ran.

# # change training settings to reflect parametsr
# TRAINING_SETTINGS['episodes'] = 250
# TRAINING_SETTINGS['max_n_steps'] = 500


N_REPS_MODEL = 160 # number of times each model is ran.

# change training settings to reflect parametsr
TRAINING_SETTINGS['episodes'] = 250
TRAINING_SETTINGS['max_n_steps'] = 500

agents =  {
    # 'QTable':QTableModel,
    'DynaQ_20':DynaQModel,
    # 'InfluenceZonesNoSheltVec':InfluenceZones,
}


agent_kwargs = {
    'QTable':dict(discount=0, learning_rate=.9, penalty_move = 0),  # penalty_move = 1e-8
    'DynaQ_20':dict(discount=0, n_planning_steps=20, penalty_move=0),   
    'InfluenceZonesNoSheltVec':dict(discount=0, predict_with_shelter_vector=False, learning_rate=.2, penalty_move=0),
}

DISCOUNT_VALUES = dict(
    none        = 0,
    vlow        = .01,
    low1        = .05,
    low2        = 0.075,
    low3        = 0.1,
    low4        = 0.25,
    low5        = 0.4,
    mid         = 0.5,
    high1       = 0.6,
    high2       = 1 - .25,
    high3       = 1 - .1,
    high4       = 1 - .075,
    high5       = 1 - .05,
    vhigh       = 1 - .01,
    max         = 1,
)

# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #
def run():
    for discount_name, discount in DISCOUNT_VALUES.items():
        for maze_name, maze in zip(('M1', 'M2', 'M3'), (PsychometricM1, PsychometricM2, PsychometricM3)):
            
            # ! skipping mazes
            if maze_name != 'M3':
                print("Skipping maze")
                continue

            logger.info(f'Training on maze: {maze_name} | Number of steps: {TRAINING_SETTINGS["episodes"]} | Max steps per episode {TRAINING_SETTINGS["max_n_steps"]}')

            for name, model in agents.items():
                # # ! skipping agents
                # if name not in ('QTable', 'DynaQ_20'):
                #     print("Skipping model")
                #     continue
                
                logger.info(f'      training agent: {name} | {N_REPS_MODEL} reps | on {maze_name} | discount: {discount}')

                # run all instances in parallel
                pool = multiprocessing.Pool(processes = 8)
                arguments = [(maze, model, name, rep, discount) for rep in range(N_REPS_MODEL)]
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
                    'play_arm':[],
                    'play_arm_sem':[],
                }
                for epn in range(TRAINING_SETTINGS['episodes']):
                    keys = [
                        ('n_steps', 'episode_length_history'),
                        ('distance_travelled', 'episode_distance_history'),
                        ('success', 'successes_history'),
                        ('play_status', 'play_status_history'),
                        ('play_steps', 'play_steps_history'),
                        ('play_arm', 'play_arm_history'),
                    ]
                    for k, v in keys:
                        try:
                            training_results[k].append(np.nanmean([th.data[v][epn] for th in training_history]))
                            training_results[k+'_sem'].append(sem([th.data[v][epn] for th in training_history], nan_policy='omit'))
                        except TypeError as e:
                            raise ValueError(f'{[th.data[v] for th in training_history]}')

                pd.DataFrame(training_results).to_hdf(f'./cache/{name}_training_on_{maze_name}_{discount_name}.h5', key='hdf')
                logger.info(f'SAVED ./cache/{name}_training_on_{maze_name}_{discount_name}.h5')

                escape_results = {  # one for each rep of the model
                    'rewards': [r[1] for r in run_results],
                    'n_steps': [r[2] for r in run_results],
                    'escape_arm': [r[3] for r in run_results],
                    'status': [r[4] for r in run_results],
                }
                pd.DataFrame(escape_results).to_hdf(f'./cache/{name}_escape_on_{maze_name}_{discount_name}.h5', key='hdf')
                logger.info(f'SAVED: /cache/{name}_escape_on_{maze_name}_{discount_name}.h5')


# ------------------------------- run instance ------------------------------- #
def run_instance(args):
    '''
        Runs a single instance and returns the results, used for
        running multiple instances in parallel.
    '''
    maze, model, name, rep, discount = args
    print(f'\n[{salmon}]Starting[/{salmon}]: Model: [b {pink}]{name}[/b {pink}] - Iteration [{blue_light}]{rep+1}/{N_REPS_MODEL}[/{blue_light}]')

    # remove duplicate parameters
    settings = TRAINING_SETTINGS.copy()
    rewards = REWARDS.copy()

    agent_kwargs[name]['discount'] = discount
    for param in agent_kwargs[name].keys():
        if param in settings.keys():
            print(f'[dim]Overring default settings value for {param}, setting it to: {agent_kwargs[name][param]}')
            del settings[param]

        # adjust rewards per model
        if param in rewards.keys():
            print(f'[dim]Overring default reward value for {param}, setting it to: {agent_kwargs[name][param]}')
            rewards[param] = agent_kwargs[name][param]

    # create an instance
    _maze = maze(rewards)
    _maze.build_graph()
    _maze.shelter_found = False
    agent = model(_maze, name=_maze.name, **settings, **agent_kwargs[name])

    # train
    agent.train(random_start=RANDOM_INIT_POS, episodes=TRAINING_SETTINGS['episodes'], test_performance=True)

    # do a play run
    try:
        status, play_steps, play_reward, escape_arm, states = _maze.play(agent, start_cell=_maze.START)
    except Exception as e:
        logger.warning(f'exception when running the model in play mode:\n{e}')
        play_steps = np.nan
        play_reward = np.nan
        escape_arm = np.nan
        status = np.nan

    print(f'\n[{green}]Finished[/{green}]: Model: [b {pink}]{name}[/b {pink}] - Iteration [{blue_light}]{rep+1}/{N_REPS_MODEL}[/{blue_light}] - play status: {status}, arm: {escape_arm}')

    agent.training_history.data['play_arm_history'] = [np.nan if x is None else (1 if x == 'right' else 0) for x in agent.training_history.play_arm_history]
    return (
        agent.training_history,
        play_reward,
        play_steps,
        escape_arm,
        status,
    )


if __name__ == "__main__":
    run()