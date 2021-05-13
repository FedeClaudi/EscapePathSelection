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

from myterial import (
    pink,
    green,
    indigo,
    orange_dark,
    blue_light,
    amber_dark,
    indigo_dark,
    indigo_light,
    salmon,
    indigo_darker,
    green_darker,
    purple,
    purple_dark,
)

from fcutils.plot.elements import plot_mean_and_error
from fcutils.plot.figure import clean_axes, save_figure
from fcutils.maths import rolling_mean


from figures.third import PsychometricM1, PsychometricM6, QTableModel, DynaQModel, InfluenceZones


from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS

'''
    Run multiple agents on a given maze to compare their performance
'''

# set logger level
logger.remove()
logger.add(sys.stdout, level="INFO")

# -------------------------- select maze and agents -------------------------- #

maze = PsychometricM1

agents =  {    
    'QTable':QTableModel,
    # 'DynaQ_5':DynaQModel,
    'DynaQ_15':DynaQModel,
    # 'DynaQ_30':DynaQModel,
    'InfluenceZones':InfluenceZones,
    # 'InfluenceZonesNoSheltVec':InfluenceZones,
}

agent_kwargs = {
    'QTable':{'learning_rate':0.3},
    'DynaQ_30':dict(n_planning_steps=30),   
    'DynaQ_15':dict(n_planning_steps=15),
    'DynaQ_5':dict(n_planning_steps=5),
    'DynaQ_1':dict(n_planning_steps=1),
    'InfluenceZones':{
        'learning_rate': .001,
        'reward_euclidean' :0,
        'reward_geodesic' :0,
        'predict_with_shelter_vector':True
    },
    'InfluenceZonesNoSheltVec':{
        'learning_rate': .9,
        'reward_euclidean' :0,
        'reward_geodesic' :0,
        'predict_with_shelter_vector':False
    },
}

colors =  {
    'QTable':green,
    'DynaQ_30':indigo_darker,
    'DynaQ_15':indigo_dark,
    'DynaQ_5':indigo,
    "InfluenceZones":purple,
    'InfluenceZonesNoSheltVec': purple_dark,
}

# -------------------------------- parameters -------------------------------- #

N_REPS_MODEL = 50 # number of times each model is ran.

N_EPISODES = 250
N_STEPS_PER_EPISODE = 500

SMOOTH_WINDOW = 25  # rolling mean smooothing window

# change training settings to reflect parametsr
TRAINING_SETTINGS['episodes'] = N_EPISODES
TRAINING_SETTINGS['max_n_steps'] = N_STEPS_PER_EPISODE

# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #
def run():
    results = {name:{
        'reward_history': [],
        'steps_history': [],
        'play_reward': [],
        'play_steps': [],
    } for name in agents.keys()}
    # run each agent
    for name, model in agents.items():
        # run all instances in parallel
        pool = multiprocessing.Pool(processes = 10)
        arguments = [(maze, model, name, rep, N_REPS_MODEL) for rep in range(N_REPS_MODEL)]
        run_results = pool.map(run_instance, arguments)

        # collate results
        results[name]['reward_history'] = [res[0] for res in run_results]
        results[name]['steps_history'] = [res[1] for res in run_results]
        results[name]['play_reward'] = [res[2] for res in run_results]
        results[name]['play_steps'] = [res[3] for res in run_results]
    return results

def run_instance(args):
    '''
        Runs a single instance and returns the results, used for
        running multiple instances in parallel.
    '''
    maze, model, name, rep, N_REPS_MODEL = args
    # print(f'\n[{salmon}]Starting[/{salmon}]: Model: [b {pink}]{name}[/b {pink}] - Iteration [{blue_light}]{rep+1}/{N_REPS_MODEL}[/{blue_light}]')

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
    agent.train(random_start=RANDOM_INIT_POS)

    # do a play run
    try:
        _, play_steps, play_reward = _maze.play(agent)
    except Exception:
        play_steps = np.nan
        play_reward = np.nan

    print(f'\n[{green}]Finished[/{green}]: Model: [b {pink}]{name}[/b {pink}] - Iteration [{blue_light}]{rep+1}/{N_REPS_MODEL}[/{blue_light}]')
    return (
        agent.training_history.max_episode_reward_history, 
        agent.training_history.episode_length_history,
        play_reward,
        play_steps
    )


# ---------------------------------------------------------------------------- #
#                                     PLOT                                     #
# ---------------------------------------------------------------------------- #

def plot(results):
    mname = maze(REWARDS).name
    # organize the data for easier plot and take comulative sum
    results = {name: (
                        to_array(data['reward_history']), 
                        to_array(data['steps_history']),
                        np.array(data['play_reward']),
                        np.array(data['play_steps']),
                    ) 
                    for name, data in results.items()}  

    f, axes = plt.subplots(ncols=2, figsize=(15, 8))
    f.suptitle(mname)
    for n, (name, res) in enumerate(results.items()):
        for data, ax in zip(res, axes):
            if len(data.shape) > 1:
                # plot mean and error over episodes
                mean = rolling_mean(np.nanmean(data, 0), SMOOTH_WINDOW)
                var = rolling_mean(scipy.stats.sem(data, 0, nan_policy='omit'), SMOOTH_WINDOW)
                plot_mean_and_error(mean, var, ax, label=name, color=colors[name], err_alpha=0.1)
            else:
                # plot as as a scatter plot
                x = np.random.normal(n, .05, size=len(data))
                ax.scatter(x, data, s=100, lw=1, edgecolors=[.3, .3, .3], color=colors[name])
    
    # mark line
    axes[0].axhline(0, lw=2, color=[.3, .3, .3])
    axes[0].axhline(REWARDS['reward_exit'], lw=2, color=[.3, .3, .3])
    axes[1].axhline(N_STEPS_PER_EPISODE, lw=2, color=[.3, .3, .3])

    # style axes
    clean_axes(f)
    f.tight_layout()
    axes[0].legend()
    axes[1].legend()

    axes[0].set(xlabel='# training episodes', ylabel='MAX reward', title='Episode rewards')
    axes[1].set(xlabel='# training episodes', ylabel='Episode length', title='Steps per episode')

    # save
    save_figure(f, f'baselines')


# ----------------------- helper functions and classes ----------------------- #


def to_array(lst):
    '''
        stacks a list of variable length lists into an array
        also takes the rolling mean of each signal
    '''
    n = len(lst)
    m = np.max([len(l) for l in lst])

    data = np.full((n,m), np.nan)
    for i, l in enumerate(lst):
        data[i, :len(l)] = l

    return data


if __name__ == "__main__":
    results = run()
    plot(results)

    print('Ready')
    plt.show()
