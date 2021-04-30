import pandas as pd
from dataclasses import dataclass

from figures.bayes import Bayes

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
        return self.trials.mouse_id.unique()

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
        return self.escape_counts_by_arm()['right']

    @property
    def nL(self):
        return self.escape_counts_by_arm()['left']

    def mice_pR(self, use_bayes=False):
        '''
            Computs the probability of going right for each mouse
            either raw or with hierarchical bayes
        '''

        if use_bayes:
            bayes = Bayes()

            # get data to fit bayes one
            n_hits_per_mouse = []
            n_trials_per_mouse = []
            for mouse in self.mice:
                trials = self.trials.loc[(self.trials.mouse_id == mouse)&(self.trials.escape_arm != 'center')]
                n_hits_per_mouse.append(len(trials.loc[trials.escape_arm == 'right']))
                n_trials_per_mouse.append(len(trials))

            # fit
            trace, means, stds = bayes.individuals_hierarchical_bayes(self.n_mice, n_hits_per_mouse, n_trials_per_mouse,
                        n_cores=10)

            return pRs
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
        counts = self.escape_numbers_by_arm()

        return {k : c/self.n_trials for k,c in counts.items()}