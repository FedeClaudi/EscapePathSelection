import sys
sys.path.append('./')


import numpy as np
import os
import pandas as pd
from nptdms import TdmsFile

from nptdms import TdmsFile
import scipy.signal as signal
from collections import OrderedDict

from fcutils.file_io.io import load_yaml
from fcutils.maths.stimuli_detection import *

from paper.dbase.ccm import run as get_matrix
from paper.dbase.utils import correct_tracking_data, get_roi_at_each_frame
from paper import paths



def load_stimuli_from_tdms(tdmspath, software='behaviour'):
        """ Takes the path to a tdms file and attempts to load the stimuli metadata from it, returns a dictionary
         with the stim times """
        # TODO load metadata
        # Try to load a .tdms
        print('\n Loading stimuli time from .tdms: {}'.format(os.path.split(tdmspath)[-1]))
        try:
            tdms = TdmsFile(tdmspath)
        except:
            raise ValueError('Could not load .tdms file: ', tdmspath)

        if software == 'behaviour':
            stimuli = dict(audio=[], visual=[], digital=[])
            for group in tdms.groups():
                for obj in tdms.group_channels(group):
                    if 'stimulis' in str(obj).lower():
                        for idx in obj.as_dataframe().loc[0].index:
                            if '  ' in idx:
                                framen = int(idx.split('  ')[1].split('-')[0])
                            else:
                                framen = int(idx.split(' ')[2].split('-')[0])

                            if 'visual' in str(obj).lower():
                                stimuli['visual'].append(framen)
                            elif 'audio' in str(obj).lower():
                                stimuli['audio'].append(framen)
                            elif 'digital' in str(obj).lower():
                                stimuli['digital'].append(framen)
                            else:
                                print('                  ... couldnt load stim correctly')
        else:
            raise ValueError('Feature not implemented yet: load stim metdata from Mantis .tdms')
        return stimuli


def load_visual_stim_log(path):
    if not os.path.isfile(path): raise FileExistsError("Couldnt find log file: ", path)
    
    try: 
        log = load_yaml(path)
    except:
        raise ValueError("Could not load: ", path)

    # Transform the loaded data into a dict that can be used for creating a df
    temp_d = {k:[] for k in log[list(log.keys())[0]]}

    for stim_i in sorted(log.keys()):
        for k in temp_d.keys():
            try:
                val = float(log[stim_i][k])
            except:
                val = log[stim_i][k]
            temp_d[k].append(val)
    return pd.DataFrame.from_dict(temp_d).sort_values("stim_count")




class ToolBox:
    def __init__(self):
        # Load paths to data folders
        self.raw_video_folder = os.path.join(
            paths.raw_data_folder, paths.raw_video_folder)

        self.raw_metadata_folder = os.path.join(
            paths.raw_data_folder, paths.raw_metadata_folder)

        self.tracked_data_folder = paths.tracked_data_folder

        self.analog_input_folder = paths.raw_ai_folder
        # os.path.join(paths.raw_data_folder, 
        #                                         paths.raw_analoginput_folder)
                                                
        self.pose_folder = paths.tracked_data_folder

    def get_behaviour_recording_files(self, session):
        raw_video_folder = self.raw_video_folder
        raw_metadata_folder = self.raw_metadata_folder

        # get video and metadata files
        videos = sorted([f for f in os.listdir(raw_video_folder)
                            if session['session_name'].lower().replace(".", '') in f.lower() and 'test' not in f
                            and '.h5' not in f and '.pickle' not in f])
        metadatas = sorted([f for f in os.listdir(raw_metadata_folder)
                            if session['session_name'].lower().replace(".", '') in f.lower() and 'test' not in f and '.tdms' in f])

        if videos is None or metadatas is None:
            raise FileNotFoundError(videos, metadatas)

        # Make sure we got the correct number of files, otherwise ask for user input
        if not videos or not metadatas:
            if not videos and not metadatas:
                return None, None
            raise FileNotFoundError('dang')
        else:
            if len(videos) != len(metadatas):
                # print('Found {} videos files: {}'.format(len(videos), videos))
                # print('Found {} metadatas files: {}'.format(
                #     len(metadatas), metadatas))
                raise ValueError(
                    'Something went wront wrong trying to get the files')

            num_recs = len(videos)
            # print(' ... found {} recs'.format(num_recs))
            return videos, metadatas


    def tdms_as_dataframe(self, tdms, to_keep, time_index=False, absolute_time=False):
        """
        Converts the TDMS file to a DataFrame

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :return: The full TDMS file data.
        :rtype: pandas.DataFrame
        """
        keys = []  # ? also return all the columns as well
        dataframe_dict = OrderedDict()
        for key, value in tdms.objects.items():
            keys.append(key)
            if key not in to_keep: continue
            if value.has_data:
                index = value.time_track(absolute_time) if time_index else None
                dataframe_dict[key] = pd.Series(data=value.data, index=index)
        return pd.DataFrame.from_dict(dataframe_dict), keys

    def open_temp_tdms_as_df(self, path, move=True, skip_df=False, memmap_dir = None):
        """open_temp_tdms_as_df [gets a file from winstore, opens it and returns the dataframe]
        
        Arguments:
            path {[str]} -- [path to a .tdms]
        """
        # Download .tdms from winstore, and open as a DataFrame
        # ? download from winstore first and then open, faster?
        if move:
            try:
                temp_file = load_tdms_from_winstore(path)
            except:
                raise ValueError("Could not move: ", path)
        else:
            temp_file = path

        print('opening ', temp_file, ' with size {} GB'.format(
            round(os.path.getsize(temp_file)/1000000000, 2)))
        bfile = open(temp_file, 'rb')
        print("  ... opened binary, now open as TDMS")

        if memmap_dir is None: memmap_dir = "M:\\"
        tdmsfile = TdmsFile(bfile, memmap_dir=memmap_dir)
        print('      ... TDMS opened')
        if skip_df:
            return tdmsfile, None
        else:
            print("          ... opening as dataframe")
            groups_to_keep = ["/'OverviewCameraTrigger_AI'/'0'", "/'ThreatCameraTrigger_AI'/'0'", "/'LDR_signal_AI'/'0'", "/'AudioIRLED_analog'/'0'", "/'WAVplayer'/'0'"]
            tdms_df, cols = self.tdms_as_dataframe(tdmsfile, groups_to_keep)
            print('              ... opened as dataframe')

            return tdms_df, cols


    def extract_behaviour_stimuli(self, aifile):
        """extract_behaviour_stimuli [given the path to a .tdms file with session metadata extract
        stim names and timestamp (in frames)]
        
        Arguments:
            aifile {[str]} -- [path to .tdms file] 
        """
        # Get .tdms as a dataframe
        tdms_df, cols = self.open_temp_tdms_as_df(aifile, move=False)

        stim_cols = [c for c in cols if 'Stimulis' in c]
        stimuli = []
        stim = namedtuple('stim', 'type name frame')
        for c in stim_cols:
            stim_type = c.split(' Stimulis')[0][2:].lower()
            if 'digit' in stim_type: continue
            stim_name = c.split('-')[-1][:-2].lower()
            try:
                stim_frame = int(c.split("'/' ")[-1].split('-')[0])
            except:
                try:
                    stim_frame = int(c.split("'/'")[-1].split('-')[0])
                except:
                    continue
            stimuli.append(stim(stim_type, stim_name, stim_frame))
        return stimuli

    def extract_ai_info(self, key, aifile):
        """
        aifile: str path to ai.tdms

        extract channels values from file and returns a key dict for dj table insertion

        """

        # Get .tdms as a dataframe
        tdms_df, cols = self.open_temp_tdms_as_df(aifile, move=True, skip_df=True)
        chs = ["/'OverviewCameraTrigger_AI'/'0'", "/'ThreatCameraTrigger_AI'/'0'", "/'AudioIRLED_AI'/'0'", "/'AudioFromSpeaker_AI'/'0'"]
        """ 
        Now extracting the data directly from the .tdms without conversion to df
        """
        key['overview_camera_triggers'] = np.round(tdms_df.object('OverviewCameraTrigger_AI', '0').data, 2)
        key['threat_camera_triggers'] = np.round(tdms_df.object('ThreatCameraTrigger_AI', '0').data, 2)
        key['audio_irled'] = np.round(tdms_df.object('AudioIRLED_AI', '0').data, 2)
        if 'AudioFromSpeaker_AI' in tdms_df.groups():
            key['audio_signal'] = np.round(tdms_df.object('AudioFromSpeaker_AI', '0').data, 2)
        else:
            key['audio_signal'] = -1
        key['ldr'] = -1  # ? insert here
        key['tstart'] = -1
        key['manuals_names'] = -1
        # warnings.warn('List of strings not currently supported, cant insert manuals names')
        key['manuals_timestamps'] = -1 #  np.array(times)
        return key