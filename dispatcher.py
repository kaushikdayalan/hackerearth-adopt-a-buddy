from sklearn import ensemble
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, verbose=2),
    "randomforest2": ensemble.RandomForestClassifier(n_estimators=400,random_state=0, n_jobs=-1, verbose=2),
    "SVC": SVC(C=1, kernel='poly', gamma='scale', decision_function_shape='ovo'),
    "xgb_breed":XGBClassifier(objective='multi:softmax', n_estimators=100, min_child_weight=8, max_depth=7, 
                             learning_rate=0.1, gamma=0.2, colsample_bytree=0.77, colsample_bylevel=0),

    "xgb_pet": XGBClassifier(objective='multi:softprob', n_estimators=700, min_child_weight=9, max_depth=11, 
                            learning_rate=0.2, gamma=0.1, colsample_bytree=0.35, colsample_bylevel=0)
}   