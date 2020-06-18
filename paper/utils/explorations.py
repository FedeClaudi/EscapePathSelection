from paper.dbase.TablesDefinitionsV4 import Explorations, Session
import pandas as pd

def get_maze_explorations(maze_design, naive=None, lights=None):
    # Get all explorations for a given maze
    query = Explorations * Session * Session.Metadata * Session.Shelter  - 'experiment_name="Foraging"' 
    
    query = (query & "maze_type={}".format(maze_design))
    if naive is not None: query = (query & "naive={}".format(naive))
    if lights is not None: query = (query & "lights={}".format(lights))

    return pd.DataFrame(query.fetch())
