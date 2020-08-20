from sklearn import ensemble


MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=300, n_jobs=-1, verbose=2),
    "randomforest2": ensemble.RandomForestClassifier(n_estimators=226,random_state=0, n_jobs=-1, verbose=2)
}   