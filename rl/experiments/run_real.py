from rich import print
from myterial import orange
import matplotlib.pyplot as plt
from loguru import logger
    
import sys 
sys.path.append('./')

from fcutils.progress import track

from rl.environment.render import Render
from experiments.experiments_lookup import EXPERIMENTS
from experiments._settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS, TRACKING_TAKE_ALL_ACTIONS
from experiments.plot import plot_q_and_trial, animate_learning



'''
    Run RL algorithms on real mazes (layout reconstructed from image) 
    and using actual tracking data from experiments on those mazes
'''


experiment = EXPERIMENTS['PsychometricM6']

for rep in range(experiment['repeats']):
    logger.info(f'Experiment rep {rep}')
    # --------------------------------- training --------------------------------- #
    # get maze
    game = experiment['maze'](REWARDS)
    print(f'Starting training of maze [salmon]{game.name}: [b {orange}]{game.description} with {len(game.empty)} states')

    # get model
    model = experiment['agent'](
            game, 
            experiment['exploration_file_name'],
            take_all_actions=TRACKING_TAKE_ALL_ACTIONS,
            trial_number=experiment['trial_number'],
            name=game.name,
            **TRAINING_SETTINGS)
    model.plot_tracking()
    print(f'Running experiment with model\n')
    print(model)

    # train
    logger.info('starting training')
    model.train(film=False)

    # ----------------------------------- PLOT ----------------------------------- #
    # render single runs
    logger.info('starting plotting simulations')
    plot_q_and_trial(game, model)

    f, ax = plt.subplots(figsize=(12, 12), tight_layout=True)
    model.render_q(ax=ax, showmaze=True, global_scale=False, cmap='Reds')

    # model.save_video()

# --------------------------- Run blocked/shortcuts -------------------------- #
if experiment['blocked'] or experiment['shortcut']:    
    if experiment['blocked']:
        TXT = 'BLOCKED'
        
        game.change_layout(game._blocked_layout)
    else:
        TXT = 'SHORTCUT'
        game.change_layout(game._shortcut_layout)
    logger.info(f'starting {TXT} animation')
    
    # change settings
    del TRAINING_SETTINGS['episodes']
    model.stop_on_error = True
    model.deterministic = True
    TRAINING_SETTINGS['exploration_rate'] = -1

    # plot once after block
    plot_q_and_trial(game, model, title=TXT)

    # create animation
    logger.info('Making animation')
    animate_learning(game, model, title=f'After {TXT} update', N=250, 
            videoname=f'{TXT}_update.mp4')

    # plot again after training
    plot_q_and_trial(game, model, title=TXT + ' with updated training')

plt.show()
