from paper.utils.explorations import get_maze_explorations


# for maze in (1, 2, 3, 4, 6):
#     print('getting explorations for maze ', maze)
#     explorations = get_maze_explorations(maze,  naive=None, lights=1)
#     explorations.to_hdf(f'D:\\Dropbox (UCL)\\Rotation_vte\\analysis_metadata\\explorations\\M{maze}_explorations.h5', key='hdf')
#     a = 1

from paper.dbase.TablesDefinitionsV4 import Explorations, Session
import pandas as pd

query = Explorations * Session * Session.Metadata * Session.Shelter
data = pd.DataFrame(query.fetch())
data = data.loc[data['experiment_name']=='Model Based V2']
data.to_hdf(f'D:\\Dropbox (UCL)\\Rotation_vte\\analysis_metadata\\explorations\\MB_explorations.h5', key='hdf')