# %%
import sys
sys.path.append(r'C:\Users\Federico\Documents\GitHub\EscapePathSelection')

from paper.helpers.mazes_stats import get_mazes
import pandas as pd
# %%
mazes =pd.DataFrame(get_mazes()).T
print(mazes)
# %%
mazes['length_ratio'] = mazes.left_path_length / (mazes.right_path_length + mazes.left_path_length)
mazes['angle_ratio'] = mazes.left_path_angle / (mazes.right_path_angle + mazes.left_path_angle)
print(mazes)
# %%
