from rich import print
from myterial import orange
import matplotlib.pyplot as plt

from loguru import logger
    
import sys
sys.path.append('./')

from fcutils.progress import track

from rl.environment.render import Render
from experiments.experiments_lookup import EXPERIMENTS
from experiments._settings import TRAINING_SETTINGS, RANDOM_INIT_POS, REWARDS
from experiments.plot import plot_results, plot_q_and_trial, animate_learning



experiment = EXPERIMENTS['PsychometricM1_SIM']

for rep in range(experiment['repeats']):
    logger.info(f'Experiment rep {rep}')

    # --------------------------------- training --------------------------------- #
    # get maze
    game = experiment['maze'](REWARDS)
    print(f'Starting training of maze [salmon]{game.name}: [b {orange}]{game.description}')

    # get model
    model = experiment['agent'](game, name=game.name, **TRAINING_SETTINGS)
    print(f'Running experiment with model\n')
    print(model)

    # train
    logger.info('starting training') 
    plt.show()
    results = model.train(random_start=RANDOM_INIT_POS, film=False, episodes=400)

    f, ax = plt.subplots(figsize=(12, 12), tight_layout=True)
    model.render_q(ax=ax, showmaze=True, global_scale=False, cmap='Reds')

    # see if the model wanted to make a video11
    try:
        model.save_video()
    except Exception:
        pass

    # ----------------------------------- PLOT ----------------------------------- #
    # render single runs
    logger.info('starting plotting simulations')
    plot_q_and_trial(game, model)

    # make clip with agent dealing with shortcut
    if experiment['blocked'] or experiment['shortcut']:
        if experiment['blocked']:
            TXT = 'BLOCKED'
            game.change_layout(game._blocked_layout)
        else:
            TXT = 'SHORTCUT'
            game.change_layout(game._shortcut_layout)

        logger.info(f'starting {TXT} animation')
        del TRAINING_SETTINGS['episodes']

        # plot once after block
        plot_q_and_trial(game, model, title=TXT)

        # create animation
        animate_learning(game, model, title=f'After {TXT} update', N=20, videoname=f'{TXT}_update.mp4')

        # plot again after training
        plot_q_and_trial(game, model, title=TXT + ' with updated training')
    

plt.show()
