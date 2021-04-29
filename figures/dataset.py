import pandas as pd
from dataclasses import dataclass

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