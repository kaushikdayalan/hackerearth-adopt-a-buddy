
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.model_selection import KFold,TimeSeriesSplit
from xgboost import XGBClassifier
import xgboost as xgb
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from sklearn.metrics import make_scorer
from functools import partial
import numpy as np 
import pandas as pd
import time
traindata = "input/breed_train_folds.csv"
df = pd.read_csv(traindata)
X_train = df.drop(['breed_category', 'kfold'],axis=1)
y_train = df['breed_category']

def objective(params):
    time1 = time.time()
    params = {
        'max_depth': int(params['max_depth']),
        'n_estimators':int(params['n_estimators']),
        'objective':params['objective'],
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'min_child_weight':int(params['min_child_weight']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        #'num_leaves': '{:.3f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 7
    count=1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    tss = TimeSeriesSplit(n_splits=FOLDS)
    score_mean = 0
    for tr_idx, val_idx in skf.split(X_train, y_train):
        clf = xgb.XGBClassifier(**params, verbose=True)
            

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)

        score = make_scorer(f1_score, average="weighted")(clf, X_vl, y_vl)
        score_mean += score
        print(f'{count} f1 - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    
    print(f'Mean f1: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)


space = {
    'max_depth': hp.quniform('max_depth', 3, 6, 1),

    'n_estimators':hp.quniform('n_estimators',100, 300, 1),

    'objective':hp.choice('objective',['multi:softmax','multi:softprob']),

    'reg_alpha':  hp.uniform('reg_alpha', 0.2, 0.5),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 0.3),
    
    'learning_rate': hp.uniform('learning_rate', 0.02, 0.04),

    'min_child_weight': hp.quniform('min_child_weight',0, 3, 1),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 0.4),

    'gamma': hp.uniform('gamma', 0, 0.3),

    #'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),


    'subsample': hp.choice('subsample', [0.3, 0.4, 0.5, 0.6]),
    
    'feature_fraction': hp.uniform('feature_fraction', 0.3, 0.6),
    
    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, 0.6)

}

if __name__ == "__main__":
    best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)
  
# Print best parameters
best_params = space_eval(space, best)

print(best)