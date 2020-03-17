# %%
from GA_simulations.classes import *

# %%
pop = Population(N_generations=5000, N_agents=100, keep_top_perc=15, p_short=1)
pop.evolve()

pop.p_short= 0
# pop.N_generations = 250
pop.evolve()


pop.plot()




# %%
