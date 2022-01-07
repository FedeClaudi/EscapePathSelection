import sys
sys.path.append('./')
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import pandas as pd

from fcutils.plot.figure import clean_axes

from figures.third import PsychometricM1, PsychometricM2, PsychometricM3, QTableTracking, DynaQTracking, InfluenceZonesTracking, fig_3_path, accepted_sessions, MAZES, Status
from figures.third.settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from figures.settings import dpi


logger.remove()
logger.add(sys.stdout, level='INFO')

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
RESULTS = dict(maze=[], model=[], results=[], escape_arms=[])
f, axes = plt.subplots(ncols=2, figsize=(16, 9))
for maze_number, (maze_name, maze) in enumerate(zip(MAZES, (PsychometricM1, PsychometricM2, PsychometricM3))):
    # if maze_name == 'M1':
    #     continue
    # else:
    print(f"Training on: {maze_name}")

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
        results, arms = [], []
        for n, session_number in enumerate(accepted_sessions[maze_name]):

            session_results, session_arms = [], []
            for i in range(10):
                logger.debug('-'*20)

                # instantiate model and maze
                _maze = maze(agent_rewards)
                _model = model(
                            _maze, 
                            maze_name,
                            take_all_actions=False,
                            trial_number=session_number,
                            name=_maze.name,
                            **agent_settings, **agent_kwargs[name])

                if i == 0:
                    logger.info(f'Maze: {maze_name} | model: {name} - session {n}/{len(accepted_sessions[maze_name])} | {len(_model.tracking)} tracking steps')


                # train
                _model.train(film=False)

                # test
                status, play_steps, play_reward, escape_arm, states = _maze.play(_model, start_cell=_maze.START)
                logger.info(f'          finished with status: {status}\n')
                session_results.append(1 if status == Status.WIN else 0)

                if status == Status.WIN:
                    session_arms.append(1 if escape_arm == 'right' else 0)

                # if name == 'InfluenceZonesNoSheltVec':
                #     break

            results.append(session_results)
            arms.append(np.mean(session_arms))

        
        RESULTS['maze'].append(maze_name)
        RESULTS['model'].append(name)
        RESULTS['results'].append(np.vstack(results))
        RESULTS['escape_arms'].append(arms)

        # axes[maze_number].hist(results, label=name, alpha=.4)

pd.DataFrame(RESULTS).to_hdf('./cache/guided_exploration.h5', key='hdf')
# axes[0].legend()
# axes[1].legend()
# plt.show()
