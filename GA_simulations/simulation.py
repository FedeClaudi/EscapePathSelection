
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import sys
sys.path.append('./')
from GA_simulations.classes import *



def run(a):
    pop = Population()
    pop.evolve()

    pop.plot()



if __name__ == "__main__":


    pool = mp.Pool(mp.cpu_count()-2)
    pool.map(run, [i for i in np.arange(50)])
    pool.close()




