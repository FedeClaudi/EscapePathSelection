import numpy as np
import numpy.random as npr
from random import choice, choices
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

from fcutils.maths.geometry import calc_distance_between_points_2d, calc_angle_between_vectors_of_points_2d
from fcutils.maths.geometry import get_random_point_on_line_between_two_points

import networkx as nx


def coin_toss(th = 0.5):
    if npr.random()>th:
        return True
    else:
        return False


# ---------------------------------------------------------------------------- #
#                                     MAZE                                     #
# ---------------------------------------------------------------------------- #
class Maze:
    def __init__(self, A, B, C_l=None, C_r=None, theta_l=None, theta_r=None, gamma_l=None, gamma_r=None):
        self.A = A
        self.B = B
        self.C_l = C_l
        self.C_r = C_r
        self.theta_l = theta_l # angle of path relative to shelter vector
        self.theta_r = theta_r
        self.gamma_l = gamma_l # length of initial path segment
        self.gamma_r = gamma_r
        self.P = None

        if C_l is not None:
            self.compute_sides()
        elif theta_l is not None:
            self.get_arms_given_thetas_and_gamma()
        self.compute_xhat()

    def get_arms_given_thetas_and_gamma(self):
        self.C_l = (np.sin(self.theta_l *self.gamma_l), np.cos(self.theta_l *self.gamma_l))
        self.C_r = (np.sin(self.theta_r*self.gamma_r), np.cos(self.theta_r*self.gamma_r))
        self.compute_sides()
    

    def compute_sides(self):
        self.AB =   calc_distance_between_points_2d(self.A, self.B)

        self.AC_l =     calc_distance_between_points_2d(self.A, self.C_l)
        self.C_lB =     calc_distance_between_points_2d(self.C_l, self.B)
        self.AC_lB =    self.AC_l + self.C_lB

        self.AC_r =     calc_distance_between_points_2d(self.A, self.C_r)
        self.C_rB =     calc_distance_between_points_2d(self.C_r, self.B)
        self.AC_rB =    self.AC_r + self.C_rB 

    def compute_xhat(self, niters=100):
        self.xhat_l = 0
        for i in range(niters):
            P = self.get_P(shortcut_on='left')
            AP = calc_distance_between_points_2d(self.A, P)
            PB = calc_distance_between_points_2d(P, self.B)
            self.xhat_l += AP + PB

            if AP + PB > self.AC_lB:
                self.visualise()
                raise ValueError
        self.xhat_l = self.xhat_l/niters
        
        self.xhat_r = 0
        for i in range(niters):
            P = self.get_P(shortcut_on='right')
            AP = calc_distance_between_points_2d(self.A, P)
            PB = calc_distance_between_points_2d(P, self.B)
            self.xhat_r += AP + PB

            if AP + PB > self.AC_rB:
                self.visualise()
                raise ValueError
        self.xhat_r = self.xhat_r/niters

    def compute_xbar(self, p_short):
        if p_short < 0 or p_short > 1: raise ValueError
        xbar_l = (1-p_short)*self.AC_lB + p_short*self.xhat_l
        xbar_r = (1-p_short)*self.AC_rB + p_short*self.xhat_r
        return xbar_l, xbar_r

    def get_P(self, shortcut_on='left'):
        # Choose between left and right arm 
        if shortcut_on == 'left':
            ac = self.AC_l
            cb = self.C_lB
            c = self.C_l
            self.shortcut_on = 'left'
        else:
            ac = self.AC_r
            cb = self.C_rB
            c = self.C_r
            self.shortcut_on = 'right'

        # See if P is in AC or CB
        segments_ratio = ac/(cb + ac)
        if coin_toss(th=segments_ratio): # P appars in CB
            self.P_in = 'CB'
            # self.P = get_random_point_between_two_points(c, self.B)
            self.P = c  # ? no shortcut if in the distal part of the arm
        else:
            self.P_in = 'AC'
            self.P = get_random_point_on_line_between_two_points(*self.A, *c)

        if np.isnan(self.P[0]) or np.isnan(self.P[1]):
            raise ValueError("Nan in P why")
        return self.P

    def visualise(self):
        if self.P is None: 
            self.get_P()
        nodes = ['A', 'B', 'C_l', 'C_r', 'P']

        edges = [('A', 'C_l'),
                    ('A', 'C_r'),
                    ('C_l', 'B'),
                    ('C_r', 'B'),
                    ('A', 'P'),
                    ('P', 'B'),
                    ]

        pos = {'A': self.A,
                'B': self.B,
                'C_l': self.C_l,
                'C_r': self.C_r,
                'P': self.P,
                }

        G=nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        nx.draw(G, with_labels=True, pos=pos)

        plt.show()




# ---------------------------------------------------------------------------- #
#                                  ENVIRONMENT                                 #
# ---------------------------------------------------------------------------- #
class Environment:
    def __init__(self, **kwargs):
        self.p_short = kwargs.pop('p_short', .1)
        self.A = kwargs.pop('A', (0, 0)) # threat pos
        self.B = kwargs.pop('B', (0, 1)) # shelter pos
        self.AB =  calc_distance_between_points_2d(self.A, self.B)

        self.N_mazes = kwargs.pop('N_mazes', 10)
        self.N_generations = kwargs.pop('N_generations', 50)
        self.N_agents = kwargs.pop('N_agents', 50)
        self.keep_top_perc = kwargs.pop('keep_top_perc', 33)

        self.x_minmax = kwargs.pop('x_minmax', 1)

        self.get_mazes()

    def get_mazes(self):
        self.mazes = []
        for i in np.arange(self.N_mazes):
            gamma = npr.uniform(0, self.x_minmax)

            theta_l = -round(np.radians(npr.uniform(1, 180)), 2)
            theta_r = round(np.radians(npr.uniform(1, 180)), 2)

            gamma_l = round(npr.uniform(self.AB*.2, self.AB*.6), 2)
            gamma_r = round(npr.uniform(self.AB*.2, self.AB*.6), 2)

            self.mazes.append(Maze(self.A, self.B, theta_l = theta_l, theta_r = theta_r, 
                                            gamma_l=gamma_l, gamma_r=gamma_r))



    def test_effect_of_pshort(self):
        f, axarr = plt.subplots(figsize=(12, 6), ncols = len(self.mazes))

        for ax, maze in zip(axarr, self.mazes):
            left, right = [], []
            for p in np.linspace(0, 1, 51):
                l, r = maze.compute_xbar(p)
                left.append(l)
                right.append(r)

            ax.plot(left, color='g', lw=2, label='left')
            ax.plot(right, color='r', lw=2, label='right')
            ax.legend()
            ax.set(xlabel='p_shortcut', ylabel='path lengths')




    def run_trial(self, agent, maze):
        # Get the agent's choice
        agent_choice, agent_expectations = agent.choose(maze)

        # Get the actual path lengths
        length_l, length_r = maze.compute_xbar(self.p_short)
        if length_l < length_r:
            correct = 'left'
        else:
            correct = 'right'

        # Evaluate outcome -> see if dead
        mindist = np.min([length_l, length_r])
        if agent_choice == 'left': 
            escape_dur = length_l
        else:
            escape_dur = length_r

        p_dead = (escape_dur / mindist) * 0.025 # death probabily is a function of how much longer was the escape
        if coin_toss(th=1-p_dead):
            agent.die()
            
        if agent_choice == correct:
            agent.corrects.append(1)
        else:
            agent.corrects.append(0)


# ---------------------------------------------------------------------------- #
#                                     AGENT                                    #
# ---------------------------------------------------------------------------- #
class Agent:
    alive = True

    mutation_std = 0.05

    def __init__(self, parent_genome={}):
        # get genes
        p_take_small_theta = parent_genome.pop('p_take_small_theta', None)
        p_take_small_geodesic = parent_genome.pop('p_take_small_geodesic', None)

        # Add to genome
        self.genome = {}
        self.add_to_genome('p_take_small_theta', p_take_small_theta)
        self.add_to_genome('p_take_small_geodesic', p_take_small_geodesic)

        self.reset()
        if not len(self.genome.keys()):
            raise ValueError

    def add_to_genome(self, gene, allel, vmin=0, vmax=1):
        if allel is None: allel = npr.uniform(vmin, vmax)
        else: allel = allel + npr.normal(0, self.mutation_std)

        if allel < vmin: allel = vmin
        elif allel > vmax: allel = vmax

        self.genome[gene] = round(allel, 2)

    def choose(self, maze):
        # Estimate the length of the two arms
        # length_l, length_r = maze.compute_xbar(self.p_short)
        length_l, length_r = maze.AC_lB, maze.AC_rB

        # Estimate the length of the two arms
        theta_l, theta_r = np.abs(maze.theta_l), np.abs(maze.theta_r)
        if theta_l is None: raise NotImplementedError

        # get rations
        len_ratio = length_l / (length_l + length_r)
        ang_ratio = theta_l / (theta_l + theta_r)

        # weight them
        if length_l < length_r:
            shortest =  0 # ?  0 = left, 1 = right
        else:
            shortest = 1

        if theta_l < theta_r:
            smallest_angle = 0
        else:
            smallest_angle = 1

        # TODO change the way the choice is made ?
        weighted_avg = (shortest * self.genome['p_take_small_geodesic'] + 
                            smallest_angle * self.genome['p_take_small_theta'])/2

        if weighted_avg > .5:
            choice = 'right'
        else:
            choice = 'left'

        return choice, (length_l, length_r)

    def reset(self):
        self.corrects = []
        self.fitness = np.nan

    def get_fitness(self):
        self.fitness = np.nanmean(self.corrects) + np.max(list(self.genome.values()))/len(self.genome.values())

    def __repr__(self):
        return f'(agent, {self.genome})'

    def __str__(self):
        return f'(agent, {self.genome})'

    def die(self):
        self.alive = False



# ---------------------------------------------------------------------------- #
#                                  POPULATION                                  #
# ---------------------------------------------------------------------------- #
class Population(Environment):
    def __init__(self, **kwargs):
        Environment.__init__(self, **kwargs)

        self.gen_num = 0
        self.agents = [Agent() for i in range(self.N_agents)]

        self.keep_top = np.int((self.N_agents/100)*self.keep_top_perc)

        self.stats = dict(
            world_p_short = [],
            agents_p_take_small_theta = [],
            agents_p_take_small_geodesic = [],
            agents_p_correct = [],
            # perc_survivors = []
        )

    def run_generation(self):
        # Change maze every 10 generations
        # if self.gen_num % 50 == 0:
        #     self.get_mazes()

        for agent in self.agents:
            for maze in self.mazes:
                self.run_trial(agent, maze)
            agent.get_fitness()
        self.gen_num += 1

    def update_population(self):
        # Keep only agents that are alive
        agents = self.agents.copy()
        self.agents = [agent for agent in agents if agent.alive]
        if not self.agents:
            raise ValueError("They all dieded")
        fitnesses = [a.fitness for a in self.agents]

        n_agents = len(self.agents)
        self.perc_survivors = n_agents / self.N_agents

        # Get stats
        self.update_stats()

        # replenish population
        new_gen = []
        while n_agents < self.N_agents:
            # choose a random parent weighted by fitness
            parent = choices(self.agents, fitnesses, k=1)[0]
            new_gen.append(Agent(parent.genome.copy())) 
            n_agents += 1
        self.agents.extend(new_gen)

        for agent in self.agents:
            agent.reset()


    def update_stats(self):
        self.stats['world_p_short'].append(self.p_short)
        self.stats['agents_p_take_small_theta'].append(np.mean([a.genome['p_take_small_theta'] 
                                                            for a in self.agents]))
        self.stats['agents_p_take_small_geodesic'].append(np.mean([a.genome['p_take_small_geodesic'] 
                                                            for a in self.agents]))
        self.stats['agents_p_correct'].append(np.mean([np.nanmean(a.corrects) 
                                                            for a in self.agents]))
        # self.stats['perc_survivors'].append(self.perc_survivors)

    def plot(self):
        f, ax = plt.subplots(figsize=(12, 6))

        for k,v in self.stats.items():
            ax.plot(v, label=k, lw=1, alpha=.7)
        ax.legend()
        ax.set(xlabel='# generations', ylabel='probability shortcut')


    def evolve(self):
        for gen_n in tqdm(range(self.N_generations)):
            self.run_generation()
            self.update_population()
