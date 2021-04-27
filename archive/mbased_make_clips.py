# %%
import pandas as pd
import os

from behaviour.videos.trials_videos import make_trials_videos

# TODO check missing videos
# TODO add mbv3 to dbase?

# %%
if __name__ == '__main__':
    
    import paper
    from paper import paths
    from paper.dbase.TablesDefinitionsV4 import Session, Stimuli, Recording

    data = pd.DataFrame((Session * Recording.FilePaths *  Stimuli & "experiment_name='Model Based V2'").fetch())
    clips_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\modelbased\\clips'
    videos_fld = 'K:\\TEMP\\video'

    sessions = data.session_name.unique()
    videos = [data.loc[data.session_name == s].overview_video.iloc[0] for s in sessions]
    videos = [os.path.join(videos_fld, os.path.split(v)[1]) for v in videos]

    stimuli = {s:data.loc[data.session_name == s].overview_frame.values for s in sessions}
    print(stimuli.keys())

    # %%
    # for video, stims in zip(videos, list(stimuli.values())):
    #     make_trials_videos(video, list(stims), save_folder = clips_fld, n_sec_pos=25)

    # %%
