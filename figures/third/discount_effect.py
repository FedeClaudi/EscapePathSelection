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

excluded = ['InfluenceZones', 'InfluenceZonesNoSheltVec', 'QTable']
cache_path = Path('/Users/federicoclaudi/Documents/Github/EscapePathSelection/cache/')


# for n, maze in enumerate(MAZES):
#     for model_n, (model, color) in enumerate(zip(MODELS, MODELS_COLORS)):

# %%
ROLLING_MEAN_WINDOW = 6
DISCOUNT_VALUES = dict(
    none        = 0.001,
    vlow        = .01,
    low1        = .05,
    low2        = 0.1,
    low3        = 0.2,
    low4        = 0.3,
    low5        = 0.4,
    mid         = 0.5,
    high1       = 0.6,
    high2       = 0.7,
    high3       = 0.8,
    high4       = 0.9,
    high5       = 0.95,
    vhigh       = 0.99,
    max         = 0.999
)

p_right = {m:{d:0 for d in DISCOUNT_VALUES.keys()} for m in MODELS}
success_rate = {m:{d:0 for d in DISCOUNT_VALUES.keys()} for m in MODELS}

for n, maze in enumerate(MAZES):
    if maze != "M3": continue

    for model_n, (model, color) in enumerate(zip(MODELS, MODELS_COLORS)):
        if model in excluded:
            continue

        for discount in DISCOUNT_VALUES.keys():
            # load data
            try:
                data_path = cache_path / f'{model}_training_on_{maze}_{discount}.h5'
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
X =  np.linspace(0, .75, len(DISCOUNT_VALUES.values()))
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
    axes[0].bar(x, successess, yerr=succ_err, width=.04, alpha=1, label=model)

    axes[1].bar(x, prights, yerr=prights_err, width=.04, alpha=1)

axes[0].legend()
axes[1].axhline(.5)

_ = axes[1].set(xticks=xticks, xticklabels=[], ylabel="p(R)", title="p(R)|discount", xlabel=f'discounts: {DISCOUNT_VALUES}')
_ = axes[0].set(xticks=xticks, xticklabels=[], ylabel="p(success)", title="success|discount",)

# %%
