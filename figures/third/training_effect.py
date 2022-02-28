# %%
import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.maths import rolling_mean
from figures.third import MODELS_COLORS, MODELS, MAZES, fig_3_path

excluded = ['InfluenceZones', 'InfluenceZonesNoSheltVec']
cache_path = Path('/Users/federicoclaudi/Documents/Github/EscapePathSelection/cache/')


# for n, maze in enumerate(MAZES):
#     for model_n, (model, color) in enumerate(zip(MODELS, MODELS_COLORS)):

# %%
ROLLING_MEAN_WINDOW = 6
N_episodes = dict(
    vlow        = 2,
    low1        = 3,
    low2        = 4,
    low3        = 5,
    low4        = 6,
    low5        = 7,
    mid         = 10,
    high1       = 50,
    
)

p_right = {m:{d:0 for d in N_episodes.keys()} for m in MODELS}
success_rate = {m:{d:0 for d in N_episodes.keys()} for m in MODELS}

for n, maze in enumerate(MAZES):
    if maze != "M3": continue

    for model_n, (model, color) in enumerate(zip(MODELS, MODELS_COLORS)):
        if model in excluded:
            continue

        for discount in N_episodes.keys():
            # load data
            try:
                data_path = cache_path / f'{model}_training_on_{maze}_{discount}_training_length.h5'
                data = pd.read_hdf(data_path, key='hdf')
            except Exception as e:
                print(e)
                continue
            
            # get success rate
            success_rate[model][discount] = (data.success.values[-1], data.success_sem.values[-1])
            p_right[model][discount] = (data.play_arm.values[-1], data.play_arm_sem.values[-1])

            # if model == "DynaQ_20": 
            #     dd = data
            #     break
            


# %%
f, axes = plt.subplots(figsize=(16, 8), ncols=2)
X =  np.linspace(0, .75, len(N_episodes.values()))
xticks = []
for n, (model, pRs) in enumerate(p_right.items()):
    x = n + X
    xticks.extend(list(x))

    try:
        prights = [pr[0] for pr in pRs.values()]
        successess = [sr[0] for sr in success_rate[model].values()]

        prights_err = [pr[1] for pr in pRs.values()]
        succ_err = [sr[1] for sr in success_rate[model].values()]
    except:
        continue

    # ax.bar(x, successess, width=.1, color="k", alpha=.5)
    axes[0].bar(x, successess, yerr=succ_err, width=.06, alpha=1, label=model)

    axes[1].bar(x, prights, yerr=prights_err, width=.06, alpha=1)

axes[0].legend()
axes[1].axhline(.5)

_ = axes[1].set(xticks=xticks, xticklabels=[], ylabel="p(R)", title="p(R)|discount", xlabel=f'discounts: {N_episodes}')
_ = axes[0].set(xticks=xticks, xticklabels=[], ylabel="p(success)", title="success|discount",)

# %%
