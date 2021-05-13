# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


from pigeon import annotate
from IPython.display import display
from PIL import Image

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent
sys.path.append(str(module_path))
sys.path.append('./')

from fcutils.video import get_cap_from_file, get_cap_selected_frame
from fcutils.progress import track
from fcutils.path import to_yaml

from paper import Trials, Recording

# %%
# get all trials
trials = pd.concat(
    [
        Trials.get_by_condition(experiment_name='Model Based V2'),
        Trials.get_by_condition(experiment_name='Model Based V3')
    ]
).reset_index()

print(len(trials))

# %%
def get_trial_frame(trial):
    rec = pd.Series((
        Recording.FilePaths & f'recording_uid="{trial.recording_uid}"'
    ).fetch1())

    path = Path(rec.overview_video)
    video_base = Path(r'Z:\swc\branco\Federico\raw_behaviour\maze\video')
    video = get_cap_from_file(video_base / path.name)

    return get_cap_selected_frame(video, trial.stim_frame + 5 * 40)


# get frames
images = {}
for i, trial in track(trials.iterrows(), total=len(trials)):
    images[trial.stimulus_uid + '_' + str(trial.stim_frame)] = get_trial_frame(trial)

# %%
def display_image(image):
    print(image)
    img = Image.fromarray(images[image]).resize((200, 200))
    display(img)


# %%
# annotate
annotations = annotate(
    images.keys(),
    options=['open', 'closed', 'skip'],
    display_fn=display_image
)

# %%
to_yaml('mb_annotations.yaml', {k:v for k,v in annotations})
# %%
