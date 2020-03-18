import sys
sys.path.append('./')
import matplotlib.pyplot as plt

from GA_simulations.classes import *

pop = Population(N_generations=2000, N_agents=200, N_mazes=500, p_short=.5)
pop.evolve()


pop.plot()
plt.show()

