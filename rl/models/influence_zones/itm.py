from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import random

from myterial import salmon, blue_grey, green_dark, orange_darker
from myterial.utils import map_color

'''
    Code implementing Instantaneous Topological Map
'''

@dataclass
class Node:
    id: int  # number to identify the node
    W: np.ndarray  # reference vector
    edges: list = field(default_factory=list) # list of neighboring nodes
    Q: dict = field(default_factory=dict) # state-action value associated with each neighboring node
    r: float = 0 # reward when at state
    e: int = 0  # indicates if the node is considered in the current topological neighborhood

    def __sub__(self, node):
        '''
            return the difference between the reference vector of two nodes
        '''
        return self.W - node.W

    def __str__(self):
        return f'{self.id} at {tuple(self.W)}'

    def __repr__(self):
        return f'{self.id} at {tuple(self.W)}'

    @property
    def V(self):
        '''
            node's value
        '''
        return np.max(list(self.Q.values())) if len(self.Q) else 0

    @property
    def has_edges(self):
        return len(self.edges)

    @property
    def neighors(self):
        return self.edges

    @property
    def target_delta(self):
        '''
            Returns a vector pointing in the direction
            of the highest valued neighbor
        '''
        best = self.edges[np.argmax([n.V for n in self.edges])]
        return best.W - self.W

    def remove_edge(self, edge_node):
        if edge_node in self.edges:
            self.edges.pop(self.edges.index(edge_node))

    def update_Q(self):
        '''
            update Q to add/remove actions to nodes
            that are/are not in the edges of this node
        '''
        edges_id = [n.id for n in self.edges]
        for eid in edges_id:
            if eid not in self.Q.keys():
                self.Q[eid] = 0.0
                
        for eid in self.Q.copy().keys():
            if eid not in edges_id:
                del self.Q[eid]


# ----------------------- Instantaneous Topological Map ---------------------- #

class ITM:
    epsilon = .2  # lerning rate
    e_max = 1.2  # 1.2  # max error before adding/removing nodes

    def __init__(self):
        # initialize nodes
        self.nodes = []
        self.next_id = 0

    @property
    def n_nodes(self):
        return len(self.nodes)

    def get_closest(self, state):
        '''
            Gets the node closest to the current state
        '''
        state = np.array(state)
        distances = np.array([np.linalg.norm(state - node.W) for node in self.nodes])
        keep = [self.nodes[n] for n,d in enumerate(distances) if d == distances.min()]
        return random.choice(keep)

    def _get_two_closest(self, state):
        '''
            Gets the two nodes closest to the state

            Arguments:
                state: np.ndarray with environment's state

            Returns
                n, s: closest and second closest Nodes (based on their vectors and input state)
        '''
        distances = np.array([np.linalg.norm(state - node.W) for node in self.nodes])
        n_idx, s_idx = np.argsort(distances)[:2]

        return self.nodes[n_idx], self.nodes[s_idx]

    def _node_idx(self, node):
        return [i for i, n in enumerate(self.nodes) if n.id == node.id][0]

    def update(self, state):
        '''
            Updates the topological map based on the input state vector.
            Algorithm:
                1. get two nodes closes to state
                2. update closest node
                3. create/delete connections to ensure delunay triangulation
                4. create/delete nodes to ensure accuracy
        '''
        if self.n_nodes < 2:
            # at first just add nodes
            self._add_node(np.array(state))
            return

        # get two closest nodes
        state = np.array(state)
        n, s = self._get_two_closest(state)

        # create connection
        if s not in n.edges:
            n.edges.append(s)
            s.edges.append(n)

        # update closest node's vector
        n.W += self.epsilon * (state - n.W)

        # remove all non-delunay edges
        for m in n.edges:
            if np.dot((n - s), (m - s)) < 0:
                n.remove_edge(m)
                m.remove_edge(n)
        
        # remove orphan nodes
        for node in self.nodes:
            if not node.has_edges:
                self.nodes.pop(self._node_idx(node))

        # add nodes
        if (
            np.linalg.norm(n.W - state) > self.e_max and
            np.dot((n.W - state), (s.W - state)) > 0
        ):
            self._add_node(state)


        # remove nodes
        if s in self.nodes:
            if np.linalg.norm(n - s)  < 0.5 * self.e_max:
                # make sure s doesn't feature as edge anywhere
                for node in self.nodes:
                    node.remove_edge(s)
                self.nodes.pop(self._node_idx(s))

    def _add_node(self, state):
        '''
            Adds a node with vector pointing at a state
        '''
        self.nodes.append(
            Node(
                self.next_id,
                state.astype(np.float32)
                )
        )
        self.next_id += 1

    def _get_highest_value_node(self, nodes=None):
        '''
            Returns the node in a list of nodes with highest V attribute
        '''
        nodes = nodes or self.nodes
        values = [n.V for n in nodes]
        return nodes[np.argmax(values)]

    def construct_neighborhoods(self, current_node):
        '''
            Constructs topological neighborhoods given a current node

            Returns
                neighborhoods: list of list of Nodes representing the  
                    topological neighborhoods constructed from a given node
        '''
        # iterate over increasinly distance neighborhoods
        current_node.e = 1
        neighborhood = [current_node]
        neighborhoods = [[current_node]]
        while neighborhood:
            # get next level
            next_neighborhood = []
            for n in neighborhood:
                for next_n in n.edges:
                    if next_n.e == 0:
                        next_n.e = 1
                        next_neighborhood.append(next_n)

            if not next_neighborhood:
                break

            # remove duplicates and keep
            # next_neighborhood = list(set(next_neighborhood))
            neighborhoods.append(next_neighborhood.copy())
            neighborhood = next_neighborhood

        # reset 'e' attributes
        for node in self.nodes:
            node.e = 0

        # check that all nodes were included
        neighborhood_nodes = [item for sublist in neighborhoods for item in sublist]
        if len(neighborhood_nodes) != self.n_nodes:
            # raise ValueError(f'ITM has {self.n_nodes} nodes but {len(neighborhood_nodes)} are in neighborhoods!')
            logger.debug(f'ITM has {self.n_nodes} nodes but {len(neighborhood_nodes)} are in neighborhoods!')

        return neighborhoods

    def visualize(self, state=None, grey=False, value=False, ax=None, annotate=False):
        '''
            plots the current map

            Arguments:
                state: current state to highlight
                grey: bool, if True nodes are nodes are plotted
                    in grey.
                value: bool. If True and grey is False nodes are colored by values
                ax: None, matplotlib axis
                annotate: if true the idx of each node is added
        '''
        if ax is None:
            f, ax = plt.subplots(figsize=(9, 9))
            ax.axis('off')
            ax.invert_yaxis()
            ax.set(title='Topological map' if not value else 'topological maps V(s)')

        minval = np.min([node.V for node in self.nodes])
        maxval = np.max([node.V for node in self.nodes])

        for node in self.nodes:
            # show node
            if grey:
                color = blue_grey
            elif value:
                if node.V:
                    color = map_color(node.V, name='Oranges', vmin=minval, vmax=maxval)
                else:
                    color = 'k'
            else:
                color = salmon
            ax.scatter(*node.W, s=100, color=color, 
                                lw=1, edgecolors=blue_grey, zorder=100)
            if annotate:
                ax.annotate(str(node.id), node.W, zorder=200, color=orange_darker, fontsize='medium')

            # plot edges
            for m in node.edges:
                ax.plot(
                    [node.W[0], m.W[0]],
                    [node.W[1], m.W[1]],
                    lw=1, 
                    color=blue_grey
                )

        # mark highest value node
        if value:
            best = self._get_highest_value_node()
            ax.scatter(*best.W, s=200, color='w', lw=1, edgecolors='k', alpha=.8)

        if state is not None:
            ax.scatter(*state, s=150, color=green_dark, lw=1, edgecolors=blue_grey, zorder=120)
        
        return ax

    def visualize_neighborhoods(self, current_node, ax=None):
        '''
            plots the current map with nodes colored by neighborhood
        '''

        # plot the whole map
        ax = self.visualize(state=current_node.W, grey=True, ax=ax)

        neighborhoods = self.construct_neighborhoods(current_node)
        for n, neighborhood in enumerate(neighborhoods):
            color = map_color(n, name='viridis', vmin=0, vmax=len(neighborhoods))
            for node in neighborhood:
                ax.scatter(*node.W, s=100, color=color, lw=1, edgecolors=blue_grey, zorder=115)
        ax.set(title='neighborhoods')
        

    def visualize_3D(self):
        '''
            Visualize the ITM by mapping node locations + values to 3D
        '''

        f = plt.figure(figsize=(9, 9))
        ax = f.add_subplot(111, projection='3d')

        for node in self.nodes:
            ax.scatter(*node.W, node.V, c=salmon, lw=1, edgecolors=blue_grey, s=200)
            ax.scatter(*node.W, 0, c='k', lw=1, edgecolors='k', s=200)
            for neighbor in node.edges:
                # plot edges
                ax.plot(
                    [node.W[0], neighbor.W[0]],
                    [node.W[1], neighbor.W[1]],
                    [node.V, neighbor.V],
                    lw=3, color=blue_grey
                )
