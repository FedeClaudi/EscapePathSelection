import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.metrics import  r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from fcutils.progress import track

from figures._data_utils import get_pR_from_trials_df

@dataclass
class Results:
    model: sm.GLM
    xtrain: pd.DataFrame
    xtest: pd.DataFrame
    ytrain: pd.DataFrame
    ytest: pd.DataFrame
    predictions: pd.Series


class GLM:
    def __init__(self, data, predictors):
        '''
            Fits a GLM with binomial link function.

            data: pd.DataFame with all data. Should be a dataframe of trials
            predictors: list of str of column names
            predicted: list of str of column names
        '''
        # normalize predictors
        X = data[predictors + ['outcomes']]
        for var in ('time_in_session',):
            X[var] = MinMaxScaler().fit_transform(X[var].values.reshape(-1,1))
        X['maze'] = data.maze
        self.X = sm.add_constant(X, prepend=False)


    @classmethod
    def from_datasets(cls, datasets):
        for data in datasets:
            angle_ratio = data.maze['left_path_angle']/ (data.maze['right_path_angle'] + data.maze['left_path_angle'])
            data.trials['angle_ratio'] = angle_ratio
            data.trials['geodesic_ratio'] = data.maze['ratio']
            data.trials['outcomes'] = [1 if arm == 'right' else 0 for arm in data.trials.escape_arm.values]
            data.trials['origin'] = [1 if arm == 'right' else 0 for arm in data.trials.origin_arm.values]
            data.trials['maze'] = data.maze['maze_name'].upper()

            fps = [30 if t.uid < 184 else 40 for i,t in data.trials.iterrows()]
            data.trials['time_in_session'] = [t.stim_frame_session / fps[n] for n, (_, t) in enumerate(data.trials.iterrows())]

        # glm_data = pd.DataFrame(glm_data)
        glm_data = pd.concat([d.trials for d in datasets]).reset_index()
        glm_data = glm_data[['maze', 'angle_ratio', 'geodesic_ratio', 'outcomes', 'origin', 'time_in_session']]

        return cls(glm_data, ['angle_ratio', 'geodesic_ratio', 'origin', 'time_in_session'])

    @staticmethod
    def get_Y_from_X(X):
        '''
            Returns a dataframe of 1 column with pR for each maze
        '''
        X = X.drop('index', axis=1)
        pr = np.zeros(len(X))
        for maze in X.maze.unique():
            maze_trials = X.loc[X.maze == maze]
            pr[maze_trials.index] = maze_trials.outcomes.mean()

        return pd.DataFrame(dict(pR=pr))

    def prepare_data(self):
        '''
            Splits the data into a random train and test sets and comptues p(R) for each
        '''
        n_samples = int(len(self.X) * .66)
        xtrain = self.X.sample(n_samples).reset_index()
        xtest = self.X[~self.X.isin(xtrain)].dropna().reset_index()

        ytrain = self.get_Y_from_X(xtrain)
        ytest = self.get_Y_from_X(xtest)

        xtrain = xtrain.drop(['index', 'maze', 'outcomes'], axis=1)
        xtest = xtest.drop(['index', 'maze', 'outcomes'], axis=1)
        
        return xtrain, xtest, ytrain, ytest

    def fit(self):
        '''
            Fits the model
        '''
        # split train test
        xtrain, xtest, ytrain, ytest = self.prepare_data()

        # fit model
        model = sm.GLM(ytrain, xtrain, family=sm.families.Binomial())
        fitted_model = model.fit()

        # predict on test
        y_hat = fitted_model.predict(xtest)

        return Results(
            fitted_model,
            xtrain,
            xtest,
            ytrain,
            ytest,
            y_hat
        )

    def fit_bootstrapped(self, repetitions=100):
        '''
            repeatedly fits the model on random subsets of the data
            and returns the R2 of the predictions and a dataframe of the model coefficients
        '''
        R2, params = [], []
        for _ in track(range(repetitions)):
            res = self.fit()

            Ypred = list(res.predictions.values)
            Y = list(res.ytest.pR)

            R2.append(r2_score(Y, Ypred))
            params.append(res.model.params)
        return R2, pd.DataFrame(params)
        