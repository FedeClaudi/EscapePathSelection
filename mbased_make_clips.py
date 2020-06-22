# %%
import pandas as pd
import os

from behaviour.videos.trials_videos import make_videos_in_parallel

import paper
from paper import paths
from paper.dbase.TablesDefinitionsV4 import Session, Stimuli, Recording


# %%
# data = pd.DataFrame((Session * Recording.FilePaths *  Stimuli & "experiment_name='Model Based V2'").fetch())
clips_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\modelbased\\clips'
videos_fld = 'K:\\TEMP\\video'

sessions = data.session_name.unique()
videos = [data.loc[data.session_name == s].overview_video.iloc[0] for s in sessions]
videos = [os.path.join(videos_fld, os.path.split(v)[1]) for v in videos]

stimuli = {s:data.loc[data.session_name == s].overview_frame.values for s in sessions}
stimuli

# %%
make_videos_in_parallel(videos, list(stimuli.values()), save_fld = clips_fld, n_sec_pos=25)