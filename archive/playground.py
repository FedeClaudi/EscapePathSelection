# %%
import numpy as np
import pandas as pd 
from scipy.integrate import odeint
from collections import namedtuple
import matplotlib.pyplot as plt
# %%
state = namedtuple('state', 'x, y, theta, v')

m = 3

def deriv(x, t):
    if t < 30:
        u = [5, 2]
    else:
        u = [1, 1]

    x = state(*x)
    dxdt = [
        x.v * np.cos(np.radians(x.theta)),
        x.v * np.sin(np.radians(x.theta)),
        (u[0]-u[1])/m,
        (u[0]+u[1])/m * (1 - np.abs((u[0] - u[1])/(u[0] + u[1])))
    ]
    return dxdt

t = np.linspace(0, 50)
x0 = state(0, 0, 0, 0)

history = odeint(deriv, x0, t)

f, axarr = plt.subplots(ncols=3, figsize=(10, 4))

axarr[0].plot(history[:, 0], history[:, 1])
axarr[0].set(xlabel='x', ylabel='y')

keys = ['theta', 'v']
for i, key in enumerate(keys):
    axarr[i+1].plot(history[:, i+2])
    axarr[i+1].set(ylabel=key)


# %%
