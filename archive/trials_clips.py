from pathlib import Path
import pandas as pd
from tqdm import tqdm

from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter

import paper
from paper import paths
from paper.dbase.TablesDefinitionsV4 import *


clips_fld = Path(r'Z:\swc\branco\Federico\Locomotion\control\behavioural_data\federico')
videos_fld = Path(r'Z:\swc\branco\Federico\raw_behaviour\maze\video')


data = pd.DataFrame((Session * Session.Metadata * Recording.FilePaths *  Stimuli).fetch())

for i, session in tqdm(data.iterrows()):
    if session.overview_video == 'not_found':
        print('video not found')
        continue

    vname = Path(session.overview_video).name
    vpath = videos_fld / vname
    if not vpath.exists():
        print('vpath not exist')
        continue

    frame = session.overview_frame
    ext = 'avi' if 'avi' in vpath.name else 'mp4'

    savepath = clips_fld / f'{vpath.stem}_{frame}.{ext}'
    if savepath.exists():
        continue


    # Open video
    videocap = get_cap_from_file(str(vpath))
    nframes, width, height, fps, _ = get_video_params(videocap)
    writer = open_cvwriter(str(savepath), w=width, h=height, framerate=fps, iscolor=True)

    n_frames_pre = 2 * fps
    n_frames_pos = 10 * fps
    get_cap_selected_frame(videocap, int(frame-1))
    for framen in tqdm(np.arange(frame-n_frames_pre, frame+n_frames_pos)):
            ret, frame = videocap.read()
            if not ret: break

            writer.write(frame.astype(np.uint8))
    writer.release()

