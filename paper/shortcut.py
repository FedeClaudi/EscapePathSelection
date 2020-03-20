# %%
import sys
sys.path.append('/Users/federicoclaudi/Documents/Github/EscapePathSelection')

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

from fcutils.file_io.io import load_yaml
from fcutils.file_io.utils import get_file_name
from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter

from behaviour.tdms.utils import get_analog_inputs_clean_dataframe
from behaviour.utilities.signals import get_times_signal_high_and_low, convert_from_sample_to_frame

import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes

# %%
# --------------------------------- Load data -------------------------------- #
notes = load_yaml(paper.paths.shortcut_notes)

sessions = ['200227_CA8755_1', '200227_CA8754_1', '200227_CA8753_1', '200227_CA8752_1',
            '200225_CA8751_1', '200225_CA848_1', '200225_CA8483_1', '200225_CA834_1',
            '200225_CA832_1', '200210_CA8491_1', '200210_CA8482_1', '200210_CA8481_1',
            '200210_CA8472_1', '200210_CA8471_1', '200210_CA8283_1']







# %%
for session in sessions:
    if not notes[session]['overall'] == 'keep': continue

    # TODO continue

# %%
