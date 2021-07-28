"""
Defines class for Successor Options and Incremental Successor options
"""
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from loguru import logger
from dataclasses import dataclass
from sklearn.cluster import KMeans

from fcutils.progress import track
from myterial import orange, green

from rl.models.successor_options.qlearner import SimpleQLearner
from rl.models.successor_options.option import Option
from rl.environment.environment import Actions


class Successor():
    def __init__(self):
        """s
        Wrapper for obtaining Successor Options
        """
        self.height = self.environment.maze.shape[0]
        self.width = self.environment.maze.shape[1]

        # create look ups
        self.state_indices = {state:n for n, state in enumerate(self.environment.cells)}
        self.states_lookup = {v:k for k,v in self.state_indices.items()}

        # create successor matrix
        self.successor = np.zeros((self.environment.n_cells, self.environment.n_cells))

    def get_successor(self, load=False, iters=int(10e6)):
        """
        Returns the sucessor representations after obtaining samples from
        uniformly random policy
        """
        learning_rate = .1
        discount = 1

        if not load:
            state = random.choice(self.environment.empty)

            for i in track(range(iters), total=iters, description='Getting successor', transient=True):
                # select an action
                action = random.choice(self.environment.actions)

                # execute each action
                next_state = self.environment.step(action)[0]
                
                # prepare stuff
                state_index = self.state_indices[state]
                next_state_index = self.state_indices[next_state]
                oneHot = np.zeros(self.environment.n_cells)
                oneHot[state_index] = 1

                # update successor representation via TD learning
                self.successor[state_index] += learning_rate * \
                    (oneHot + discount * self.successor[next_state_index] - self.successor[state_index])
                state = next_state

            # save to file
            np.save('./data/successor.npy', self.successor)
        else:
            self.successor = np.load('./data/successor.npy')

    def get_subgoals(self, n_clusters):
        """
        Cluster the successor representations into clusters using k-centroids
        """
        #  Get SR for valid states
        keep_idxs = [i for i, cell in enumerate(self.environment.cells) if cell in self.environment._cells]
        valid_SR = self.successor[keep_idxs, :]

        # check SR is complete
        if np.any(valid_SR.max(axis=1) == 0):
            # raise ValueError('Valid successor should not be 0')
            logger.warning('Valid successor should not be 0')
        
        # cluster: get indices in valid SR
        centroid_indices, cluster_labels = cluster(valid_SR, k=n_clusters)
        
         # get centroids
        # centroids_states = [self.environment.empty[idx] for idx in centroid_indices]
        centroids_states = [self.states_lookup[idx] for idx in centroid_indices]
        self.centroids = [Centroid(self.state_indices[state], state) for state in centroids_states]

        # plot
        # self.plot_clusters(cluster_labels)

    def learn_option_policies(self, episodes=10, steps=1e5, load=False):
        '''
            Builds a policy for each subgoal by using simple
            Q learning with a pseudo reward function
        '''
        if not load:
            # initialize options with no Q
            _options = [Option(n, centroid.state, centroid.state_index, np.zeros(1)) 
                            for n, centroid in enumerate(self.centroids)]
            
            # Learn each option's policy
            self.options = []
            for n, option in enumerate(_options):
                logger.debug(f'Building subgoal policy {n+1}/{len(self.centroids)}')
                reward_vector = self.successor[option.subgoal_state_index]  # reward for each state

                # learn and save option and policy
                learner = SimpleQLearner(self.environment, 
                        reward_vector,
                        self.state_indices,
                        option, 
                        n_actions=self.environment.n_actions)
                option.Q = learner.train(episodes=episodes, steps=int(steps))

                option.save()
                self.options.append(option)
        else:
            self.options = [Option.load(n) for n, _ in enumerate(self.centroids)]

    def plot_subgoals(self):
        f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        f.suptitle('Subgoals')

        successor = self.successor.copy()

        for n, (ax, centroid) in enumerate(zip(axarr.flatten(), self.centroids)):
            # plot the maze
            ax.imshow(self.environment.maze, cmap="binary")

            # plot the successor representation of the subgoal
            mat = np.reshape(successor[centroid.state_index], (self.height, self.width))
            mat[mat==0] = np.nan  # to plot as transparent
            ax.imshow(mat.T, cmap='viridis', origin='lower', alpha=.9)

            # mark centroid
            ax.scatter(centroid.state[0], centroid.state[1], zorder=100, color='r')
            ax.set(title=f'Subgoal {n}', xticks=[], yticks=[])
            ax.invert_yaxis()

    def plot_clusters(self, cluster_labels):
        '''
            cluster_labels: labels each empty cell with the cluster it belongs to
        '''

        clusters = [(state, label) for state, label in zip(self.environment._cells, cluster_labels)]

        f, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 9))
        f.suptitle('Clusters')

        for n, ax in enumerate(axes.flatten()):
            ax.imshow(self.environment.maze, cmap='binary')

            # get states that belong to this cluster
            cluster = np.zeros_like(self.environment.maze)
            for (x,y), label in clusters:
                if label == n:
                    cluster[y, x] = 1
            cluster[cluster == 0] = np.nan
            ax.imshow(cluster, cmap='viridis', vmin=0, vmax=2)

            ax.set(title=f'Cluster {n}', xticks=[], yticks=[])

    def plot_options(self):
        '''
            Plot each option and Q policy associated with it
        '''
        states_lookup = {v:k for k,v in self.state_indices.items()}

        # create a figure to plot the policies on
        f, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 9))

        for (ax, option) in zip(axes.flatten(), self.options):
            # plot maze on ax to show policy for this subgoal
            ax.imshow(self.environment.maze, cmap='binary')
            ax.scatter(*option.subgoal_state, zorder=103, s=15, color='r')
            ax.set(title=f'Q policy for subgoal {option.number}', xticks=[], yticks=[])
                
            for n in range(option.Q.shape[0]):
                # get the values of each action at each state
                state = states_lookup[n]
                if state not in self.environment.empty:
                    continue

                q = option.Q[n]
                if q.max() <= 0:
                    ax.scatter(*state, color=orange, s=50, zorder=102)
                else:
                    # plot action with highest value
                    action = np.argmax(q)

                    dx = 0
                    dy = 0
                    if action == Action.MOVE_LEFT:
                        dx = -0.3
                    if action == Action.MOVE_RIGHT:
                        dx = +0.3
                    if action == Action.MOVE_UP:
                        dy = -0.3
                    if action == Action.MOVE_DOWN:
                        dy = 0.3
                        
                    ax.arrow(*state, dx, dy, color=green, head_width=0.4, head_length=0.1)


@dataclass
class Centroid:
    state_index: int
    state: tuple


def cluster(data, k=3):
    """
    K-means clustering using sklearn
    Returns cluster assignments

    data: SR for each state. Size: n_valid_states x maze_size**2
    """
    # fit Kmeans model to get clusters
    model = KMeans(n_clusters=k, n_init=100, algorithm='full')
    cluster_labels = list(model.fit_predict(data))
    centers = [np.argmax(c) for c in model.cluster_centers_]
    return centers, cluster_labels
