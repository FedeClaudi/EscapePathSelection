import sys
sys.path.append('./')
import matplotlib.pyplot as plt

from GA_simulations.classes import *

pop = Population()
pop.evolve()

pop.plot()
plt.show()

