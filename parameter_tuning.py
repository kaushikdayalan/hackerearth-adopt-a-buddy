import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import dispatcher
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

#MODEL = "xgb_breed"
breed_TrainingData = 'input/breed_train_folds.csv'
pet_TrainingData = 'input/pet_train_folds.csv'


breed_params = {
    
}

pet_params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "colsample_bylevel":[0,1],
    "n_estimators": [50, 70, 10, 100, 200, 400, 700, 500, 1000],
    "max_depth": [10,11, 12, 13, 14, 15, 16, 17, 20],
    "min_child_weight": [6, 7, 8, 9, 10, 12, 14, 15],
    "gamma": [0.0, 0.1, 0.2, 0.35, 0.4, 0.45, 0.5, 0.55, 0.7],
    "colsample_bytree": [0.3, 0.35, 0.4, 0.45, 0.5],
    "objective": ['multi:softmax', 'multi:softprob']
}

if __name__ == "__main__":
    df = pd.read_csv(pet_TrainingData)
    ytrain = df.pet_category.values
    df = df.drop(['length(m)','height(cm)','pet_category','breed_category','kfold'], axis=1)

    scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
    clf = XGBClassifier()
    random_search = RandomizedSearchCV(clf, param_distributions=pet_params, n_iter=5, 
    scoring=scorer, n_jobs=-1, verbose=3, cv=5)
    random_search.fit(X= df, y=ytrain)

    print(random_search.best_estimator_)
    print()
    print(random_search.best_params_)
    print()
    print(random_search.best_score_)