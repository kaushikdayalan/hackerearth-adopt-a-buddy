from sklearn import ensemble
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=500, n_jobs=-1,max_depth=10,verbose=1),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, verbose=2),
    "randomforest2": ensemble.RandomForestClassifier(max_depth=15, n_jobs=-1, n_estimators=500),
    "xgb_breed":XGBClassifier(max_depth=2, learning_rate=0.01,n_estimators=400,
                                min_child_weight=1),
    "xgb_pet": XGBClassifier(max_depth=5,colsample_bylevel=0.5,reg_alpha=0.1, reg_lambda=0.1,learning_rate=0.1, n_estimators=500),
    "xgb": XGBClassifier()
}   
