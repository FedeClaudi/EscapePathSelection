import pandas as pd
import numpy as np
from dataclasses import dataclass

from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as get_orientation

from paper import Tracking

from figures.bayes import Bayes
from figures._data_utils import register_in_time

@dataclass
class DataSet:
    name: str
    trials: pd.DataFrame

    def __repr__(self):
        return f'''
            DATASET: {self.name}
            -------------
            n trials: {self.n_trials}
            n sessions: {self.n_sessions}
            n mice: {self.n_mice}

            mazes: {self.mazes}
            experiments: {self.experiments}
        '''

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        else:
            try:
                return getattr(self.__dict__['trials'], name)
            except AttributeError:
                raise AttributeError(f'{self.name} does not have attribute {name}')

    @property
    def n_trials(self):
        return len(self.trials)

    @property
    def n_mice(self):
        return len(self.trials.mouse_id.unique())

    @property
    def mice(self):
        return list(self.trials.mouse_id.unique())

    @property
    def sessions(self):
        return list(self.trials.session_name.unique())

    @property
    def n_sessions(self):
        return len(self.trials.uid.unique())

    @property
    def mazes(self):
        return self.trials.maze_type.unique()

    @property
    def experiments(self):
        return self.trials.experiment_name.unique()

    @property
    def pR(self):
        return self.escape_probability_by_arm()['right']

    @property
    def pL(self):
        return self.escape_probability_by_arm()['left']

    @property
    def nR(self):
        return self.escape_numbers_by_arm()['right']

    @property
    def nL(self):
        return self.escape_numbers_by_arm()['left']

    @property
    def L(self):
        return self.trials.loc[self.trials.escape_arm == 'left']


    @property
    def R(self):
        return self.trials.loc[self.trials.escape_arm == 'right']

    def _get_mice_ntrials(self):
        '''
            For each mouse in the datasets returns the number of trials
            and the number of trials for each arm
        '''
        # get data to fit bayes one
        n_R_per_mouse = np.zeros(self.n_mice)
        n_trials_per_mouse = np.zeros(self.n_mice)
        for n, mouse in enumerate(self.mice):
            trials = self.trials.loc[(self.trials.mouse_id == mouse)&(self.trials.escape_arm != 'center')]
            n_R_per_mouse[n] = len(trials.loc[trials.escape_arm == 'right'])
            n_trials_per_mouse[n] = len(trials)

        n_L_per_mouse = n_trials_per_mouse - n_R_per_mouse
        return n_trials_per_mouse, n_L_per_mouse, n_R_per_mouse

    def grouped_pR(self):
        '''
            Returns the grouped p(R) for all mice in
            the dataset using bayes 
        '''
        bayes = Bayes()
        _, _, _, _, _, prange, _ = bayes.grouped_bayes_analytical(self.n_trials, self.nR)

        return prange

    def mice_pR(self, use_bayes=False):
        '''
            Computs the probability of going right for each mouse
            either raw or with hierarchical bayes
        '''
        if use_bayes:
            bayes = Bayes()
            n_trials_per_mouse, _, n_R_per_mouse = self._get_mice_ntrials()

            # fit
            trace, means, stds = bayes.individuals_hierarchical_bayes(self.n_mice, n_R_per_mouse, n_trials_per_mouse,
                        n_cores=10)

            return means
        else:
            pRs = []
            for mouse in self.mice:
                trials = self.trials.loc[(self.trials.mouse_id == mouse)&(self.trials.escape_arm != 'center')]
                pRs.append(len(trials.loc[trials.escape_arm == 'right']) / len(trials))

            return pRs
            

    def clean(self):
        '''
            Removes bad trials which end away from the shelter because
            of tracking errors
        '''
        to_drop = []
        for i, trial in self.trials.iterrows():
            if trial.y[-1] < 135:
                to_drop.append(i)

        self.trials.drop(to_drop, inplace=True)

    def escape_numbers_by_arm(self, ignore_center=True):
        '''
            For each available escape arm it computes the nuymber of
            escapes along it.
        '''
        if ignore_center:
            trials = self.trials.loc[self.trials.escape_arm != 'center']
        else:
            trials = self.trials

        arms = trials.escape_arm.unique()
        counts = {}

        for arm in arms:
            arm_trials = trials.loc[trials.escape_arm==arm]
            counts[arm] = len(arm_trials)

        return counts

    def escape_probability_by_arm(self, ignore_center=True):
        '''
            For each available escape arm it computes the probability of
            escapes along it.
        '''
        counts = self.escape_numbers_by_arm(ignore_center=ignore_center)

        return {k : c/self.n_trials for k,c in counts.items()}


    def get_orientations_on_T(self, n_samples=50, catwalk_only=True):
        '''
            Returns the tracking and orientation data
            from each trial but only for while the mouse is on T
        '''
        data = dict(x=[], y=[], speed=[], orientation=[], escape_arm=[])
        R, L = [], []
        for i, trial in self.trials.iterrows():
            if catwalk_only and trial.y[0] > 50:
                continue
                    
            start = trial.stim_frame
            end = trial.out_of_t_frame

            # get orientation while on T
            body = pd.DataFrame(Tracking.BodyPart & f'recording_uid="{trial.recording_uid}"' & 'bpname="body"').iloc[0]
            snout = pd.DataFrame(Tracking.BodyPart & f'recording_uid="{trial.recording_uid}"' & 'bpname="snout"').iloc[0]

            orientation = get_orientation(body.x[start:end], body.y[start:end],
                                            snout.x[start:end], snout.y[start:end])
            if trial.escape_arm == 'right':
                R.append(orientation)
            elif trial.escape_arm == 'left':
                L.append(orientation)
            else:
                continue
            
            data['x'].append(body.x[start:end])
            data['y'].append(body.y[start:end])
            data['speed'].append(body.speed[start:end])
            data['escape_arm'].append(trial.escape_arm)


        '''
            Each trial has a different duration, to take averages we rescale them
            all to have the same duration
        '''
        L_aligned = register_in_time(L, n_samples)
        R_aligned = register_in_time(R, n_samples)
        lcount, rcount = 0, 0
        for side in data['escape_arm']:
            if side == 'left':
                data['orientation'].append(L_aligned[:, lcount])
                lcount += 1
            else:
                data['orientation'].append(R_aligned[:, rcount])
                rcount += 1

        return pd.DataFrame(data)