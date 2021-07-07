# %%
# %%
import sys
from pathlib import Path
import os
from rich import print

from myterial import salmon

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')



from figures.first import M1, M2, M3, M4,M6,  fig_1_path
from figures.glm import GLM
from figures._plot_utils import generate_figure
from figures.settings import dpi

datasets = (M1, M2, M3, M4, M6)

# %%
glm = GLM.from_datasets(datasets)
R2, params = glm.fit_bootstrapped(repetitions=50)
# 

ax = generate_figure()
ax.hist(R2, bins=20)
_ = ax.set(xlabel='R2', ylabel='count')
ax.figure.savefig(fig_1_path / 'panel_F_Rsquared.eps', format='eps', dpi=dpi)

print(f'[{salmon}]Params means', params.mean(), sep='\n')
print(f'[{salmon}]\n\nParams STDs', params.std(), sep='\n')

# %%
res = glm.fit()
print(res.model.summary())

print(res.predictions)
print(res.ytest)


# %%
