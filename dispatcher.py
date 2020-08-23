from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

level0 = list()
level0.append(('randomforest', RandomForestClassifier()))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('svm', SVC()))
level0.append(('xgb', XGBClassifier()))
#level0.append(('mlp', MLPClassifier()))
level1 = LogisticRegression(max_iter=3000) # define meta learner model
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

xgb_breed_params = {'bagging_fraction': 0.5449642187248955, 'colsample_bytree': 0.35707713635800326, 
'feature_fraction': 0.4409366526692125, 'gamma': 0.17314624058753417, 'learning_rate': 0.03301290960878166, 
'max_depth': 6, 'min_child_weight': 2, 'n_estimators': 230, 'objective': 'multi:softprob', 
'reg_alpha': 0.40289213598255347, 'reg_lambda': 0.07809374095505581, 'subsample': 0.3}

xgb_pet_params=  {'max_depth': 27, 'n_estimators': 748, 'objective': 'multi:softprob', 
'gamma': 0.255, 'subsample': 0.80, 'min_child_weight': 1, 'reg_alpha': 0.065, 
'reg_lambda': 0.042, 'learning_rate': 0.085, 'colsample_bytree': 0.745, 
'feature_fraction': 0.424, 'bagging_fraction': 0.411}


MODELS = {
    "randomforest_breed": ensemble.RandomForestClassifier(criterion='gini', max_depth=8.0,
                                                max_features=0.6630566343969866, n_estimators=203,verbose=3),
    
    "randomforest_pet": ensemble.RandomForestClassifier(criterion="gini", max_depth=17.0, 
                                        max_features=0.6378796341184251, n_estimators=539,verbose=3),

    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, verbose=2),

    "xgb_breed":XGBClassifier(**xgb_breed_params),
    "xgb_pet":XGBClassifier(**xgb_pet_params),
    
    "xgb": XGBClassifier(),

    "xgb_tune":XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=6,
                            min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,reg_alpha=0.005,
                            objective= 'multi:softprob', seed=27),
    "stacking": model
}   
