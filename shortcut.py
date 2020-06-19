# %%
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import pandas as pd
import os
from math import sqrt
from scipy.signal import resample

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.colors import *
from fcutils.plotting.plot_elements import plot_mean_and_error
from fcutils.plotting.plot_distributions import plot_fitted_curve, plot_kde
from fcutils.plotting.colors import desaturate_color
from fcutils.file_io.io import load_yaml


from paper.dbase.TablesDefinitionsV4 import Session, Stimuli, TrackingData
import paper
from paper import paths
from paper.trials import TrialsLoader
from paper.helpers.mazes_stats import get_mazes
from paper.utils.misc import resample_list_of_arrayes_to_avg_len



# %%
# TODO populate sessions
# TODO populate tracking data
# TODO Get tracking data and trials and plot stuff

# ---------------------------- Load notes and data --------------------------- #

notes_path = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\shortctu\\notes.yml'
notes = load_yaml(notes_path)

keep_sessions = [k[:-2] for k,n in notes.items() if n['overall']=='keep']
print(keep_sessions)

print("Loading data")
# Get the tracking data after the stimulus for each trial
for session in keep_sessions:
    sname = f'{session}_1'
    stimuli = (Session * Stimuli * TrackingData * TrackingData.BodyPartData 
                            & f'session_name="{session}"' & "bpname='body'")

    note_trials = notes[sname]['trials']

    if len(stimuli) != len(note_trials):
        raise ValueError(f'Wrong number of stimuli {session}')
    else:
        print(f'All good {session}')




# %%
