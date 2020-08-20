from sklearn import ensemble
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, verbose=2),
    "randomforest2": ensemble.RandomForestClassifier(n_estimators=400,random_state=0, n_jobs=-1, verbose=2),
    "xgb_breed":XGBClassifier(max_depth=2, subsample=0.8, n_estimators=500, 
                                learning_rate=0.02, min_child_weight=2, random_state=5),
    "xgb_pet": XGBClassifier(max_depth=3, min_child_weight=2, random_state=5)
}   