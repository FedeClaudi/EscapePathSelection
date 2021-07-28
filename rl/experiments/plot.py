import matplotlib
matplotlib.use('MacOSX')

import matplotlib.pyplot as plt
from celluloid import Camera
from loguru import logger

from rl.environment.render import Render
from experiments._settings import TRAINING_SETTINGS


def plot_results(results, MIN_N_STEPS):
    ''' plots results of training '''
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, tight_layout=True, figsize=(12, 8))

    ax1.plot(results.win_history, lw=2, color=[.3, .3, .3])
    ax1.set_ylabel("win rate")

    ax2.plot(results.n_steps_history, lw=2, color=[.3, .3, .3], label='# steps')
    ax2.axhline(MIN_N_STEPS, color='salmon', label='min # steps')
    ax2.legend()
    ax2.set_ylabel("N steps")
    ax2.set_xlabel("episode")

def plot_q_and_trial(game, model, title=''):
    '''
        Creates a figure with 4 plots
            1. euclidean distance from shelter at all points
            2. geodesic distance from shelter at all point
            3. q values for each action and each state
            4. an example trial
    '''
    f, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 9), sharex=True, sharey=True)
    f.suptitle(title)
    axes = axes.flatten()
    
    logger.info('Rendering maze with euclidean and geodeisc distances')
    game._render_maze(distance='euclidean', ax=axes[0])
    game._render_maze(distance='geodesic', ax=axes[1])
    model.render_q(ax=axes[2])

    logger.info('Rendering PLAY')
    game.render(Render.MOVES, ax=axes[3])
    _, _, reward, _ = game.play(model, start_cell=game.START, ax=axes[3])
    logger.debug(f'Finished PLAY with reward: {reward}')
    
    axes[3].get_figure().canvas.draw()
    axes[3].get_figure().canvas.flush_events()

    return f, axes

def animate_learning(game, model, title='', N=200, videoname='animation.mp4'):
    # create a camera and a new figure
    f, ax = plt.subplots(figsize=(8, 8))
    
    camera = Camera(f)

    for i in range(N):
        logger.debug(f'Animation {i+1}/{N}')

        # do 1 episode of training
        game.render(Render.MOVES, ax=ax)
        model.train(
            random_start=False, episodes=1,
        )

        # update figure
        ax.text(-.5, -.7, f'{i+1} iterations - {title}', fontsize=20, color='k')
        game.render_q(model, ax=ax)

        camera.snap()
    
    # save animation
    animation = camera.animate()
    logger.info(f'Saving animation at: {videoname}')
    animation.save(videoname)