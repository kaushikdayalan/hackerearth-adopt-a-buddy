import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import dispatcher
import joblib
from sklearn.model_selection import RandomizedSearchCV

MODEL = "xgb_pet"
breed_TrainingData = 'input/breed_train_folds.csv'
pet_TrainingData = 'input/pet_train_folds.csv'


breed_params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [1, 2,3, 4, 5],
    "min_child_weight": [3, 4, 5, 6, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.6, 0.65, 0.7, 0.75, 0.8]
}

pet_params = {
    "learning_rate": [0.01, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [13, 14, 15, 16, 17, 20],
    "min_child_weight": [6, 7, 8, 9, 10],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.35],
    "colsample_bytree": [0.3, 0.35, 0.4, 0.45, 0.5]
}

if __name__ == "__main__":
    df = pd.read_csv(pet_TrainingData)
    ytrain = df.pet_category.values
    df = df.drop(['length(m)','height(cm)','pet_category','breed_category','kfold'], axis=1)

    scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
    clf = dispatcher.MODELS[MODEL]
    random_search = RandomizedSearchCV(clf, param_distributions=pet_params, n_iter=5, 
    scoring=scorer, n_jobs=-1, verbose=3, cv=5)
    random_search.fit(X= df, y=ytrain)

    print(random_search.best_estimator_)
    print()
    print(random_search.best_params_)