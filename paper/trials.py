import sys
sys.path.append('./')

import paper
from paper.dbase.TablesDefinitionsV4 import *
from paper.helpers.bayes import Bayes

import pandas as pd
import numpy as np

"""
    Class and utility functions to load trials from a set of experiments. 

"""

class TrialsLoader(Bayes):
    max_duration_th = 9 # ? only trials in which the mice reach the shelter within this number of seconds are considered escapes (if using escapes == True)
    shelter_location = [500, 850]

    naive_lookup = {0: "experienced", 1:"naive", -1:"nan"}
    lights_lookup = {0: "off", 1:"on", 2:"on_trials", 3:"on_exploration", -1:"nan"}

    def __init__(self, naive=None, lights=None, escapes_dur=True, shelter=True, 
                    tracking=None, catwalk=None, experiment_name=None):

        """
            Keyword Arguments:
                maze_design {[int]} -- [Number of maze of experiment] (default: {None})
                naive {[int]} -- [1 for naive only mice] (default: {None})
                lights {[int]} -- [1 for light on only experiments] (default: {None})
                escapes_dur {bool} -- [If true only trials in which the escapes terminate within the duraiton th are used] (default: {True})
                catwalk {bool} -- [If true only trials for experiments with the catwalk are returned] (default: {True})
                tracking{[bool, str]} -- [if not None the tracking for the trials is returned with the trials. If "threat" only the threat tracking is returned]
                experiment_name {[str]} -- [pass the name of an experiment as a string to select only data for that experiment]
        """

        Bayes.__init__(self)

        self.naive = naive
        self.lights = lights
        self.escapes_dur = escapes_dur
        self.shelter = shelter
        self.tracking = tracking
        self.catwalk = catwalk
        self.experiment_name = experiment_name

        self.datasets = {}

    # ---------------------------------------------------------------------------- #
    #                                 GETTING DATA                                 #
    # ---------------------------------------------------------------------------- #

    def load_psychometric(self):
        self.datasets = dict(
                maze1 =  self.load_trials_by_condition(maze_design=1),
                maze2 =  self.load_trials_by_condition(maze_design=2),
                maze3 =  self.load_trials_by_condition(maze_design=3),
                maze4 =  self.load_trials_by_condition(maze_design=4),
                maze6 =  self.load_trials_by_condition(maze_design=6),
                            )

    def load_trials_by_condition(self, maze_design=None):
        """[Given a number of criteria, load the trials that match these criteria]		
        """
        # Get all trials from the AllTrials Table
        if self.tracking is not None:
            if self.tracking == "all":
                if self.experiment_name is None:
                    all_trials = pd.DataFrame((Session * Trials * Trials.TrialTracking * Trials.TrialSessionMetadata.proj(ename='experiment_name') & "escape_duration > 0").fetch())
                else:
                    all_trials = pd.DataFrame((Session * Trials * Trials.TrialTracking * Trials.TrialSessionMetadata.proj(ename='experiment_name')\
                        & "escape_duration > 0" & "experiment_name='{}'".format(self.experiment_name)).fetch())
            elif self.tracking == "threat":
                if self.experiment_name is None:
                    all_trials = pd.DataFrame((Session * Trials * Trials.ThreatTracking * Trials.TrialSessionMetadata.proj(ename='experiment_name') & "escape_duration > 0").fetch())
                else:
                    all_trials = pd.DataFrame((Session * Trials * Trials.ThreatTracking * Trials.TrialSessionMetadata.proj(ename='experiment_name')\
                        & "escape_duration > 0" & "experiment_name='{}'".format(self.experiment_name)).fetch())
            else:
                raise ValueError("tracking parameter not valid")
        else:
            all_trials = pd.DataFrame(Trials.fetch())

        if self.escapes_dur:
            all_trials = all_trials.loc[all_trials.escape_duration <= self.max_duration_th]

        # Remove trials with negative escape duration [placeholders]
        all_trials = all_trials.loc[all_trials.escape_duration > 0]

        # Get the sessions that match the criteria and use them to discard other trials
        sessions = self.get_sessions_by_condition(maze_design)
        ss = set(sorted(sessions.uid.values))
        trials = all_trials.loc[all_trials.uid.isin(ss)]

        # Augment the trials dataframe
        trials = self.augment_trials_dataframe(trials)

        return trials

    def get_sessions_by_condition(self, maze_design, df=True):
        """ Query the DJ database table AllTrials for the trials that match the conditions """
        data = Session * Session.Metadata * Session.Shelter  - 'experiment_name="Foraging"'


        if maze_design == 1 and self.experiment_name != 'shortcut':
            # Some sessions fromshortcut are mistakenly labelled as having maze=1 instad of maze=8
            data -= 'experiment_name="shortcut"'

        if maze_design is not None:
            data = (data & "maze_type={}".format(maze_design))

        if self.naive is not None:
            data = (data & "naive={}".format(self.naive))

        if self.lights is not None:
            data = (data & "lights={}".format(self.lights))

        if self.shelter is not None:
            if self.shelter:
                data = (data & "shelter={}".format(1))
            else:
                data = (data & "shelter={}".format(0))
                
        if not len(data): print("Query didn't yield any results!")


        if df:
            return pd.DataFrame((data).fetch())
        else: 
            return data

    def augment_trials_dataframe(self, trials):
        """
            Adds stuff like time of stim onset in seconds etc...
        """
        # Get the stim time in seconds
        trials['stim_time_s'] = trials['stim_frame'] / trials['fps']

        # Get the stim number in each session
        session_stim_number = []
        sessions = set(trials.session_name)
        for sess in sessions:
            sess_trials = trials.loc[trials.session_name == sess]
            session_stim_number.extend([i for i in np.arange(len(sess_trials))])
        trials['trial_num_in_session'] = session_stim_number
        return trials

    def get_datasets_sessions(self):
        self.datasets_sessions = {ds:{'session_name':trs.session_name.unique(), 'uid':trs.uid.unique()} 
                    for ds, trs in self.datasets.items()}
        return self.datasets_sessions

    def remove_change_of_mind_trials(self):
        """On some trials mice go on one arm first, then change their mind and take the other,
        this function removes this kind of trials"""
        datasets = self.datasets.copy()
        for ds, trials in datasets.items():
            goodids = []
            for i, trial in trials.iterrows():
                if trial.escape_arm == "left":
                    if np.max(trial.body_xy[:, 0]) > 550: # moue went left and right
                        continue
                elif trial.escape_arm == "right":
                    if np.min(trial.body_xy[:, 0]) < 450: # mouse went right and left
                        continue
                goodids.append(trial.stimulus_uid)
            
            good_trials = self.datasets[ds].loc[self.datasets[ds].stimulus_uid.isin(goodids)]
            self.datasets[ds] = good_trials



    # ---------------------------------------------------------------------------- #
    #                                   ANALYSIS                                   #
    # ---------------------------------------------------------------------------- #

    def get_binary_trials_per_dataset(self, datasets=None, ignore_center=True):
        # ? datasets should be a dict whose keys should be a list of strings with the names of the different datasets to be modelled
        # ? the values of datasets should be a a list of dataframes, each specifying the trials for one dataset (e.g. maze design) and the session they belong to

        if datasets is None: datasets = self.datasets

        # Parse data
        # Get trials
        trs = {k:[] for k in datasets.keys()}
        for dataset, df in datasets.items():
            sessions = sorted(set(df.uid.values))
            for sess in sessions:
                df = df.loc[df.escape_arm != "center"]
                trs[dataset].append([1 if "right" in arm.lower() else 0 for arm in df.loc[df.uid==sess].escape_arm.values])

        # Get hits and number of trials
        hits = {c:[np.sum(t2) for t2 in t] for c, t in trs.items()}
        ntrials = {c:[len(t2) for t2 in t] for c,t in trs.items()}
        p_r = {c: [h/n for h,n in zip(hits[c], ntrials[c])] for c in hits.keys()}
        n_mice = {c:len(v) for c,v in hits.items()}
        return hits, ntrials, p_r, n_mice, trs

    def grouped_bayes_by_dataset_analytical(self, datasets=None):
        if datasets is None: datasets = self.datasets

        results = {"dataset":[], "alpha":[], "beta":[], "mean":[],      
                    "median":[], "sigmasquared":[], "prange":[],
                    "distribution":[],}
        hits, ntrials, p_r, n_mice, trials = self.get_binary_trials_per_dataset(datasets)

        for (cond, h), n in zip(hits.items(), ntrials.values()):
            res = self.grouped_bayes_analytical(np.sum(n), np.sum(h))
            results['dataset'].append(cond)
            results['alpha'].append(res[0])
            results['beta'].append(res[1])
            results['mean'].append(res[2])
            results['median'].append(res[3])
            results['sigmasquared'].append(res[4])
            results['prange'].append(res[5])
            results['distribution'].append(res[6])

        return pd.DataFrame(results)

    def individuals_bayes_by_dataset_hierarchical(self, datasets=None, **kwargs):
        if datasets is None: datasets = self.datasets

        results = dict(dataset=[], traces=[], means=[], stds=[])

        hits, ntrials, p_r, n_mice, trials = self.get_binary_trials_per_dataset(datasets)

        for cond in hits.keys():
            nhits, ntrs, nmice = hits[cond], ntrials[cond], n_mice[cond]

            res = self.individuals_hierarchical_bayes(nmice, nhits, ntrs, **kwargs)

            results['dataset'].append(cond)
            results['traces'].append(res[0])
            results['means'].append(res[1])
            results['stds'].append(res[2])

        return pd.DataFrame(results)

        