# %%
import sys
sys.path.append('./')

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fcutils.file_io.utils import listdir, get_file_name
from fcutils.file_io.io import load_json
from fcutils.plotting.plot_distributions import plot_kde

# %%
base_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\GAMODELLING'

subdir = 'nagents_100_pshort_0_nmazes'
gen_dir = 'gen_500'


subdirs = [d for d in listdir(base_fld) if subdir in d]

data = np.array([np.load(listdir(os.path.join(sd, gen_dir))[0])[0] for sd in subdirs])

data = pd.DataFrame(dict(left_length=data[:, 0],
                        right_length=data[:, 1],
                        left_theta=data[:, 2],
                        right_theta=data[:, 3],
                        ))


params = load_json(os.path.join(subdirs[0], 'params.json'))

print("Params:")
params


# %%
f, ax = plt.subplots(figsize=(12, 12))

for col in data.columns:
    # ax.hist(data[col], bins=10, label=col)
    plot_kde(ax, data=data[col], label=col)

ax.legend()


# %%
