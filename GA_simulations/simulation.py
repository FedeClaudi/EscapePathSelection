import sys
sys.path.append('./')
import matplotlib.pyplot as plt

from GA_simulations.classes import *


pop = Population(N_generations=1800, N_agents=200, N_mazes=10, p_short=0)
pop.evolve()

pop.p_short= 1
pop.evolve()


pop.plot()
plt.show()

# TODO change the way agents make a choice between options
# TODO replace agens with NN, genome is weights

# TODO agent's fitness is weighted by their genome