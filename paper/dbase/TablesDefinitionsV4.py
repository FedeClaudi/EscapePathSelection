
import datajoint as dj
import pandas as pd
import numpy as np
from scipy.signal import resample
import time

from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)
from fcutils.maths import rolling_mean


import sys
sys.path.append('./')
import paper
from paper.dbase.TablePopulateFuncsV4 import *
from paper import schema
from paper.dbase.utils import get_roi_at_each_frame



# ---------------------------------------------------------------------------- #
#                                     MOUSE                                    #
# ---------------------------------------------------------------------------- #

@schema
class Mouse(dj.Manual):
    definition = """
        # Mouse table lists all the mice used and the relevant attributes
        mouse_id: varchar(128)                        # unique mouse id
        ---
        strain:   varchar(128)                        # genetic strain
        sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
    """

    # ? population method is in populate_database



# ---------------------------------------------------------------------------- #
#                                    SESSION                                   #
# ---------------------------------------------------------------------------- #

@schema
class Session(dj.Manual):
    definition = """
    # A session is one behavioural experiment performed on one mouse on one day
    uid: smallint     # unique number that defines each session
    session_name:           varchar(128)        # unique name that defines each session - YYMMDD_MOUSEID
    mouse_id: varchar(128)                        # unique mouse id
    ---
    date:                   date                # date in the YYYY-MM-DD format
    experiment_name:        varchar(128)        # name of the experiment the session is part of 
    """
    # ? population method is in populate_database

    class Metadata(dj.Part):
        definition = """
        -> Session
        ---
        maze_type: int  # maze design id
        naive: int      # was the mouse naive
        lights: int     # light on, off, na, or part on part off
        """

    class Shelter(dj.Part):
        definition = """
        -> Session
        ---
        shelter: int  # 0 if there is no shelter 1 otherwise
        """


    def get_experiments_in_table(self):
        return set(Session.fetch("experiment_name"))

    def get_by(self, maze=None, naive=None, lights=None, fetch=False):
        data = self * self.Metadata

        if maze is not None:
            data = data & f'maze_type="{maze}"'

        if naive is not None:
            data = data & f'naive="{naive}"'

        if lights is not None:
            data = data & f'lights="{lights}"'

        if fetch:
            return pd.DataFrame(data.fetch())
        else:
            return data

    @staticmethod
    def fps(session_uid):
        '''Returns the fPS of a session given its uid'''
        if session_uid < 184: 
            return 30
        else:
            return 40

    @staticmethod
    def session_metadata(uid):
        '''Returns all metadta given a session uid'''
        return Session * Session.Metadata * Session.Shelter & f'uid="{uid}"'

# ---------------------------------------------------------------------------- #
#                                MAZE COMPONENTS                               #
# ---------------------------------------------------------------------------- #
@schema
class MazeComponents(dj.Imported):
    definition = """
    # stores the position of each maze template for one experiment
    -> Session
    ---
    s: longblob  # Shelter platform template position
    t: longblob  # Threat platform
    p1: longblob  # Other platforms
    p2: longblob
    p3: longblob
    p4: longblob
    p5: longblob
    p6: longblob
    b1: longblob  # Bridges
    b2: longblob
    b3: longblob
    b4: longblob
    b5: longblob
    b6: longblob
    b7: longblob
    b8: longblob
    b9: longblob
    b10: longblob
    b11: longblob
    b12: longblob
    b13: longblob
    b14: longblob
    b15: longblob
    """

    def make(self, key):
        new_key = make_templates_table(key)
        if new_key is not None:
            self.insert1(new_key)






# ---------------------------------------------------------------------------- #
#                           COMMON COORDINATES MATRIX                          #
# ---------------------------------------------------------------------------- #
@schema
class CCM(dj.Imported):
    definition = """
    # stores common coordinates matrix for a session
    -> Session
    camera: varchar(32)
    ---
    maze_model:             longblob        # 2d array with image used for correction
    correction_matrix:      longblob        # 2x3 Matrix used for correction
    alignment_points:       longblob        # array of X,Y coords of points used for affine transform
    top_pad:                int             # y-shift
    side_pad:               int             # x-shift
    """
    def make(self, key):
        make_commoncoordinatematrices_table(self, key)






# ---------------------------------------------------------------------------- #
#                                   RECORDING                                  #
# ---------------------------------------------------------------------------- #
@schema
class Recording(dj.Imported):
    definition = """
        # Within one session one may perform several recordings. Each recording has its own video and metadata files
        recording_uid: varchar(128)
        -> Session
        ---
        software:           enum('behaviour', 'mantis')
        ai_file_path:       varchar(256)
    """ 
    
    def make(self, key):
        make_recording_table(self, key)

    def get_experiments_in_table(self):
        return set((Session * Recording).fetch("experiment_name"))


    class FilePaths(dj.Part):
        definition = """
            # stores a reference to all the relevant files
            -> Recording
            ---
            ai_file_path:           varchar(256)        # path to mantis .tdms file with analog inputs and stims infos
            overview_video:         varchar(256)        # path to the videofile
            threat_video:           varchar(256)        # path to converted .mp4 video, if any, else is same as video filepath    
            overview_pose:          varchar(256)        # path to .h5 pose file
            threat_pose:            varchar(256)        # path to .h5 pose file
            visual_stimuli_log:     varchar(256)        # path to .yml file with the log of which visual stimuli were delivered
        """

    def make_paths(self):
        fill_in_recording_paths(self)

    class AlignedFrames(dj.Part):
        definition = """
            # Alignes the overview and threat camera frames to facilitate alignement
            -> Recording
            ---
            overview_frames_timestamps:     longblob
            threat_frames_timestamps:       longblob 
            aligned_frame_timestamps:       longblob
        """

    def make_aligned_frames(self):
        fill_in_aligned_frames(self)

    @staticmethod
    def recording_n_frames(recording_uid):
        ''' gets the number of frames int he recording
            by looking at the tracking data '''
        tracking = pd.DataFrame(Tracking * Tracking.BodyPart & f'recording_uid="{recording_uid}"' & 'bpname="body"')
        return len(tracking.iloc[0].x)


# ---------------------------------------------------------------------------- #
#                                    STIMULI                                   #
# ---------------------------------------------------------------------------- #
@schema
class Stimuli(dj.Imported):
    definition = """
        # Store data about the stimuli delivered in each recording
        -> Recording
        stimulus_uid:       varchar(128)      # uniquely identifying ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
        ---
        overview_frame:     int             # frame number in overview camera (of onset)
        overview_frame_off: int
        duration:           float                   # duration in seconds
        stim_type:          varchar(128)         # audio vs visual
        stim_name:          varchar(128)         # name 
    """

    sampling_rate = 25000

    class VisualStimuliLogFile(dj.Part): # ? This is populated in the make method
        definition = """
            -> Stimuli
            ---
            filepath:       varchar(128)
        """

    class VisualStimuliMetadata(dj.Part):  # ? This is populated separately
        definition = """
            -> Stimuli
            ---
            stim_type:              varchar(128)    # loom, grating...
            modality:               varchar(128)    # linear, exponential. 
            time:                   varchar(128)    # time at which the stim was delivered
            units:                  varchar(128)    # are the params defined in degrees, cm ...
    
            start_size:             float       
            end_size:               float
            expansion_time:         float
            on_time:                float
            off_time:               float
    
            color:                  float
            background_color:        float
            contrast:               float
    
            position:               blob
            repeats:                int
            sequence_number:        float           # sequential stim number in the session
         """

    def insert_placeholder(self, key):
        key['stimulus_uid'] 		= key['recording_uid']+'_0'
        key['duration']  			= -1
        key['stim_type'] 			= 'nan'
        key['overview_frame'] 		= -1
        key['overview_frame_off'] 	= -1
        key['stim_name'] 			= "nan"
        self.insert1(key)

    def make(self, key):
        make_stimuli_table(self, key)

    def make_metadata(self):
        make_visual_stimuli_metadata(self)	
            

    @staticmethod
    def get_by_sessions(sessions):
        '''
            Given a query with sessions data returns the relevant entries
            in the Stimuli table correcting for the appropriate FPS conversions
        '''
        data = pd.DataFrame(Stimuli & sessions & 'duration>0')
        data['fps'] = [30 if s.uid < 184 else 40 for i,s in data.iterrows()]
        data['frame'] = np.array(data['overview_frame'] / data['fps'] * 40).astype(np.int64)

        return data



# ---------------------------------------------------------------------------- #
#                                 TRACKING DATA                                #
# ---------------------------------------------------------------------------- #
@schema
class TrackingData(dj.Imported):
    experiments_to_skip = ['Lambda Maze', 'PathInt2 Close', "Foraging"]

    bodyparts = ['snout', 'neck', 'body', 'tail_base',]
    skeleton = dict(head = ['snout', 'neck'], body_upper=['neck', 'body'],
                body_lower=['body', 'tail_base'], body=['tail_base', 'neck'])

    definition = """
        # store dlc data for bodyparts and body segments
        -> Recording
        camera: 		varchar(32)
        ---
    """

    class BodyPartData(dj.Part):
        definition = """
            # stores X,Y,Velocity... for a single bodypart
            -> TrackingData
            bpname: varchar(128)        # name of the bodypart
            ---
            tracking_data: longblob     # pandas dataframe with X,Y,Speed,Dir of mvmt
            x: longblob
            y: longblob
            likelihood: longblob
            speed: longblob
            direction_of_mvmt: longblob
            
        """

    class BodySegmentData(dj.Part):
        definition = """
            # Store orientaiton, ang vel..
            -> TrackingData
            segment_name: varchar(128)
            ---
            bp1: varchar(128)
            bp2: varchar(128)
            orientation: longblob
            angular_velocity: longblob
            likelihood: longblob
        """
    
    def make(self, key):
        make_trackingdata_table(self, key)


    def get_experiments_in_table(self):
        return set((TrackingData() * Recording() * Session()).fetch("experiment_name"))


@schema
class Tracking(dj.Imported):
    '''
        tracking data but for all sessions fps=40 and units
        are in physical coordinates.
    '''
    ignored_experiments = ('PathInt', 'noshelter m1', 'shortexploration m1', 'narrowbridge m5', 'Foraging', 
                'Lambda maze', 'TwoArmsLong Maze', 'FourArms Maze', 'PathInt2 Close', 'ModelBased', 
                'ModelBasedV2', 'ModelBasedV3'
        )
    cm_per_px = 0.22  # each px = cm
    definition = """
        -> TrackingData
    """

    class BodyPart(dj.Part):
        definition = """
            # stores X,Y,Velocity... for a single bodypart
            -> Tracking
            bpname: varchar(128)        # name of the bodypart
            ---
            x: longblob
            y: longblob
            speed: longblob
            roi: longblob  # which maze component is the bodypart on
        """
    def get_roi_at_each_frame(self, key, xy):
        # Get position of maze templates - and shelter
        rois = pd.DataFrame((MazeComponents & key).fetch())
        del rois['uid'], rois['session_name'], rois['mouse_id']

        # Calcualate in which ROI the body is at each frame - and distance from the shelter
        roi_at_each_frame = get_roi_at_each_frame(None, key['recording_uid'], xy, dict(rois))
        rois_ids = {p:i for i,p in enumerate(rois.keys())}  # assign a numeric value to each ROI
        return np.array([rois_ids[r] for r in roi_at_each_frame])
        

    def make(self, key):
        '''
            get data from tracking data and convert to fps and physical size
        '''
        exp = (Session & key).fetch1('experiment_name')
        if exp in self.ignored_experiments:
            return

        self.insert1(key)
        time.sleep(1)
    
        data = pd.DataFrame(TrackingData * TrackingData.BodyPartData & key)

        for i, tracking in data.iterrows():	
            if tracking.uid < 184:
                fps = 30
            else:
                fps = 40

            # smooth
            xy = pd.DataFrame(dict(x=tracking['x'], y=tracking['y']))
            xy = xy.interpolate(method='linear', axis=0, limit_direction='backward').values 
            xy = np.vstack(
                [
                    rolling_mean(xy[:, 0], int(fps / 8)),
                    rolling_mean(xy[:, 1], int(fps / 8)),
                ]
            ).T

            # change to 40 fps
            rois = tracking['tracking_data'][:, -1]
            if fps == 30:
                n_secs = len(xy) / 30
                n_samples = int(np.ceil(n_secs * 40))

                xy = resample(xy, n_samples)

                # get maze component at each frame
                rois = self.get_roi_at_each_frame(key, xy)

            # convert to physical coordinates
            # get cm to pixel ratio including CCM
            xy *= self.cm_per_px 

            # get speed trace in cm/s
            speed = get_speed_from_xy(xy[:, 0], xy[:, 1]) * 40

            bpkey = key.copy()
            bpkey['x'] = xy[:, 0]
            bpkey['y'] = xy[:, 1]
            bpkey['speed'] = speed
            bpkey['roi'] = rois
            bpkey['bpname'] = tracking.bpname
            Tracking.BodyPart.insert1(bpkey)
            time.sleep(1)



# ---------------------------------------------------------------------------- #
#                                 EXPLORATIONS                                 #
# ---------------------------------------------------------------------------- #
@schema
class Explorations(dj.Imported):
    definition = """
        -> Session
        ---
        start_frame: int
        end_frame: int
        body_tracking: longblob
        maze_roi: longblob
        duration_s: float
        distance_covered: float
        fps: int
    """

    def make(self, key):
        if key['uid'] < 184: 
            fps = 30
        else: 
            fps=40

        # Get session's recording
        recs = (Recording & key).fetch("recording_uid")
        if not np.any(recs):
            print('skipping no recs') 
            return

        # Get the first stimulus of the session
        recs_lengs = []
        first_stim = []
        rec_n = 0
        while not first_stim:
            first_stim = list((Stimuli & key & f"recording_uid='{recs[rec_n]}'" & "duration != -1").fetch("overview_frame"))
            if first_stim: break
            rec_n += 1
            if rec_n == len(recs):
                print('skipping no stims')
                return # ! no stimuli for that session
        
        # Get comulative frame numbers and concatenated tracking
        trackings = []
        n_frames = []
        for rec in recs:
            try:
                trk = (TrackingData * TrackingData.BodyPartData & key & f"recording_uid='{rec}'" & "bpname='body'").fetch("tracking_data")[0]
            except:
                print('skipping something went wrong')
                return
            n_frames.append(trk.shape[0])
            trackings.append(trk)
        cum_n_frames = np.cumsum(n_frames)

        # Get concatenated frame number of first stim
        first_stim = first_stim[0]-1
        if rec_n:
            first_stim += cum_n_frames[rec_n-1] -1

        # Get tracking up to first stim
        tracking = np.vstack(trackings)
        like = (TrackingData * TrackingData.BodyPartData & key & f"recording_uid='{recs[0]}'" & "bpname='body'").fetch("likelihood")[0]
        start = np.where(like > 0.9999)[0][0] # The first frame is when DLC finds the mouse
        start += 15 * fps # adding 15 seconds to allow for the mouse to be placed on the maze

        if start>first_stim or first_stim>tracking.shape[0]: raise ValueError

        tracking = tracking[start:first_stim, :]

        # Put evreything together and insert
        duration = tracking.shape[0]/fps
        d_covered = np.nansum(tracking[:, 2])
        maze_roi = tracking[:, -1]

        key['body_tracking'] = tracking
        key['start_frame'] = start
        key['end_frame'] = first_stim
        key['maze_roi'] = maze_roi
        key['duration_s'] = duration
        key['distance_covered'] = d_covered
        key['fps'] = fps

        self.insert1(key)
        time.sleep(1)








# ---------------------------------------------------------------------------- #
#                                    TRIALS                                    #
# ---------------------------------------------------------------------------- #
@schema
class Trials(dj.Imported):
    ignored_experiments = ('PathInt', 'noshelter m1', 'shortexploration m1', 'narrowbridge m5', 'Foraging', 
                'Lambda maze', 'TwoArmsLong Maze', 'FourArms Maze', 'PathInt2 Close', 'ModelBased', 
                'ModelBasedV2', 'ModelBasedV3'
        )
    definition = """
        -> Stimuli
        ---	
        out_of_shelter_frame: int # ? all frames wrt to recording frames
        at_threat_frame: int
        stim_frame: int              
        out_of_t_frame: int
        at_shelter_frame: int

        stim_frame_session: int  # stim frame wrt session frames count
        experiment_name: varchar(128)

        escape_duration: float        # duration in seconds
        time_out_of_t: float

        x: longblob  # tracking of mouse body
        y: longblob  # tracking of mouse body
        speed: longblob  # tracking of mouse body
        roi: longblob
        distance_travelled: longblob
        last_roi: float

        escape_arm: enum('left', "center", "right") 
        origin_arm:  enum('left', "center", "right")        
    """

    def _insert_placeholder(self, key):
        key['out_of_shelter_frame'] = -1
        key['at_threat_frame'] = -1
        key['stim_frame'] = -1
        key['out_of_t_frame'] = -1
        key['at_shelter_frame'] = -1

        key['stim_frame_session'] = -1
        key['experiment_name'] = 'placeholder'

        key['escape_duration'] = -1
        key['time_out_of_t'] = -1
        key['escape_arm'] = 'center'
        key['origin_arm'] = 'center'

        key['x'] = -1
        key['y'] = -1
        key['speed'] = -1
        key['roi'] = -1
        key['last_roi'] = -1
        key['distance_travelled'] = -1
        

        self.insert1(key)

    def make(self, key):
        make_trials_table(self, key)


    @staticmethod
    def get_by_condition(maze_design=None, naive=None, lights=None, shelter=None, experiment_name=None, df=True, escape_duration=None):
        '''
            Select trials based on some restrictions
        '''
        data = Session.Metadata * Session.Shelter * Trials  - 'experiment_name="Foraging"'
        data = data - 'experiment_name="placeholder"'

        if maze_design == 1 and experiment_name != 'shortcut':
            # Some sessions fromshortcut are mistakenly labelled as having maze=1 instad of maze=8
            data -= 'experiment_name="shortcut"'

        if experiment_name is not None:
            data = data & f'experiment_name="{experiment_name}"'

        if escape_duration is not None:
            data = data - f'escape_duration > {escape_duration}'

        if maze_design is not None:
            data = (data & "maze_type={}".format(maze_design))

        if naive is not None:
            data = (data & "naive={}".format(naive))

        if lights is not None:
            data = (data & "lights={}".format(lights))

        if shelter is not None:
            data = (data & "shelter={}".format(1 if shelter else 0))

           
        if df:
            trials = pd.DataFrame(data)
            return Trials.remove_change_of_mind_trials(trials)
        else: 
            return data

    @staticmethod
    def remove_change_of_mind_trials(trials):
        '''
            Removes trials in whic the mice went on one side first then changed their mind.
        '''

        goodids = []
        for i, trial in trials.iterrows():
            if trial.escape_arm == "left":
                if np.max(trial.x) > 135: # moue went left and right
                    continue
            elif trial.escape_arm == "right":
                if np.min(trial.x) < 85: # mouse went right and left
                    continue
            goodids.append(trial.stimulus_uid)
        
        return trials.loc[trials.stimulus_uid.isin(goodids)]

    @staticmethod
    def remove_placeholders():
        '''
            Removes place holder table entries to clean up the table
        '''
        placeholders = Trials & 'experiment_name="placeholder"'

        print(f'Found {len(pd.DataFrame(placeholders).uid.unique())} sessions in place holder entries.')
        print(placeholders)

        # Trials().delete(placeholders)

if __name__ == "__main__": 
    Explorations().populate(display_progress=True)

#     # Trials.remove_placeholders()
