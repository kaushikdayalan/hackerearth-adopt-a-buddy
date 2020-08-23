from skopt.space import space
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import metrics
import numpy as np
import pandas as pd
from functools import partial
from skopt import gp_minimize
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from sklearn import preprocessing
from sklearn import utils

train_data = "input/pet_train_folds.csv"

def optimize(params, x, y):
    model = RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest = x[test_idx]
        ytest = y[test_idx]
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc  = metrics.f1_score(ytest, preds,average="weighted")
        accuracies.append(fold_acc)
    
    return (-1.0 * np.mean(accuracies))

if __name__ == "__main__":
    df = pd.read_csv(train_data)
    X = df.drop(['pet_category','kfold'],axis=1).values
    y = df.pet_category.values

    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth",2,20, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 1)),
        "criterion": hp.choice("criterion",['gini','entropy']),
        "max_features": hp.uniform("max_features",0.01, 1),
    }

    optimization_function = partial(optimize, x=X,y=y)
    trial = Trials()
    result = fmin(fn=optimization_function,space= param_space,algo=tpe.suggest,max_evals=15,trials=trial)

    print(result)