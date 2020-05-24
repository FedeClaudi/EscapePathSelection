# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from paper.dbase.TablesDefinitionsV4 import Explorations



# %%
explorations = pd.DataFrame(Explorations().fetch())

# %%
plt.figure()

plt.plot(e.body_tracking[:, 0], e.body_tracking[:, 1])

# %%
