import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import dispatcher
import joblib
FOLD = 4
MODEL = "randomforest2"
TrainingData = 'input/train_folds.csv'

FOLD_MAPPING = {
    0:[1, 2, 3, 4],
    1:[0, 2, 3, 4],
    2:[0, 1, 3, 4],
    3:[0, 1, 2, 4],
    4:[0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TrainingData)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.pet_category.values
    yvalid = valid_df.pet_category.values

    train_df = train_df.drop(['pet_category','kfold'], axis=1)
    valid_df = valid_df.drop(['pet_category','kfold'], axis=1)
    
    valid_df = valid_df[train_df.columns]

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict(valid_df)
    print(metrics.f1_score(yvalid, preds,average='weighted'))
    joblib.dump(clf, f"models/{MODEL}_pet_{FOLD}.pkl")