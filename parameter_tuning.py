import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import dispatcher
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import dispatcher
import train


breed_TrainingData = 'input/breed_train_folds.csv'
pet_TrainingData = 'input/pet_train_folds.csv'
MODEL = "xgb_tune"

param_test1 = {
 'max_depth':range(3,10,2), ## best = 5
 'min_child_weight':range(1,6,2)  ## best = 1
}
param_test2 = {
 'max_depth':[4,5,6],  ## best = 4
 'min_child_weight':[0, 1, 2] ## best = 3
}

param_test2b = {
 'min_child_weight':[6,8,10,12]
}
# max_depth 6 min child 2
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}



if __name__ == "__main__":
    df = pd.read_csv(breed_TrainingData)
    ytrain = df.breed_category.values
    df = df.drop(['breed_category','kfold'], axis=1)
    scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
    
    clf = dispatcher.MODELS[MODEL]

    gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test7, scoring=scorer,n_jobs=4, cv=5,verbose=3)

    gsearch1.fit(df, ytrain)

    train.train_model(gsearch1.best_estimator_)

    print(gsearch1.best_estimator_)
    print()
    print(gsearch1.best_params_)
    print()
    print(gsearch1.best_score_)