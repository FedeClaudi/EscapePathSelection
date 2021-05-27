from sklearn import svm as svm_models
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.metrics import  r2_score
from sklearn.preprocessing import StandardScaler

@dataclass
class Performance:
    TP: int
    TN: int
    FP: int
    FN: int
    accuracy: float

    @property
    def tot(self):
        return self.TP + self.TN + self.FP + self.FN

@dataclass
class Results:
    model: svm_models.SVC
    xtrain: pd.DataFrame
    xtest: pd.DataFrame
    ytrain: pd.DataFrame
    ytest: pd.DataFrame
    predictions: pd.Series
    performance: Performance



class SVM:
    def __init__(self, X, Y):
        self.X = pd.DataFrame(
            StandardScaler().fit_transform(X),
            index=X.index,
            columns=X.columns)
        self.Y = Y

    def new_instance(self):
        return svm_models.SVC(kernel='linear', degree=3, gamma='auto', C=1.0)

    @classmethod
    def from_datasets(cls, datasets):
        for data in datasets:
            data.trials = data.trials[data.trials.escape_arm != 'center']
            angle_ratio = data.maze['left_path_angle']/ data.maze['right_path_angle']
            data.trials['angle_ratio'] = angle_ratio
            data.trials['geodesic_ratio'] = data.maze['ratio']
            data.trials['outcomes'] = [1 if arm == 'right' else 0 for arm in data.trials.escape_arm.values]
            data.trials['origin'] = [1 if arm == 'right' else 0 for arm in data.trials.origin_arm.values]
            data.trials['maze'] = data.maze['maze_name'].upper()

        data = pd.concat([d.trials for d in datasets]).reset_index()
        X = data[['angle_ratio', 'geodesic_ratio']]
        Y = data[['outcomes']]

        return cls(X,Y)

    def performance_measure(self, Y, Yhat):
        '''
            Computes the number of true positives, true negatives,
            false positives, false negatives and the accuracy of the model.
        '''
        Y = list(Y.ravel())
        Yhat = list(Yhat.ravel())

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(Yhat)): 
            if Y[i]==Yhat[i]==1:
                TP += 1
            if Yhat[i]==1 and Y[i]!=Yhat[i]:
                FP += 1
            if Y[i]==Yhat[i]==0:
                TN += 1
            if Yhat[i]==0 and Y[i]!=Yhat[i]:
                FN += 1
                
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        return Performance(TP, TN, FP, FN, ACC)

    def fit(self):
        '''
            Fits the model
        '''
        # split train test
        xtrain, xtest, ytrain, ytest = train_test_split(self.X, self.Y, test_size=.33)
        fitted = self.new_instance()
        fitted.fit(xtrain, ytrain.values.ravel())

        yhat = fitted.predict(xtest)
        performance = self.performance_measure(ytest.values.T, yhat)

        return Results(fitted, xtrain, xtest, ytrain, ytest, yhat, performance)


if __name__ == '__main__':
    import sys
    sys.path.append('./')

    import matplotlib.pyplot as plt
    import numpy as np
    from figures.first import M1, M2, M3,M4, M6

    datasets =  (M1, M2, M3, M6)
    svm = SVM.from_datasets(datasets)
    res = svm.fit()
    print(f'r2: {r2_score(res.ytest.outcomes, res.predictions):.3f}')


    f, ax = plt.subplots()


    noise = np.random.normal(0, .1, size=len(res.xtest))
    noise2 = np.random.normal(0, .1, size=len(res.xtest))
    color = ['g' if p==m else 'r' for p,m in zip(res.ytest.outcomes, res.predictions)]
    ax.scatter(res.xtest.angle_ratio + noise, res.xtest.geodesic_ratio + noise2, alpha=.4, c=color)
    ax.scatter(res.model.support_vectors_[:, 0], res.model.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

    a = 1
