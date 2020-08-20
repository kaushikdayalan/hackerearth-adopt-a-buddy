import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import dispatcher
import joblib

FOLD = 0
MODEL = "xgb_breed"
breed_TrainingData = 'input/breed_train_folds.csv'
pet_TrainingData = 'input/pet_train_folds.csv'
FOLD_MAPPING = {
    0:[1, 2, 3, 4],
    1:[0, 2, 3, 4],
    2:[0, 1, 3, 4],
    3:[0, 1, 2, 4],
    4:[0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(breed_TrainingData)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.breed_category.values
    yvalid = valid_df.breed_category.values

    train_df = train_df.drop(['length(m)','height(cm)','pet_category','breed_category','kfold'], axis=1)
    valid_df = valid_df.drop(['length(m)','height(cm)','pet_category','breed_category','kfold'], axis=1)
    
    valid_df = valid_df[train_df.columns]
    print(f"Model: {MODEL}, FOLD: {FOLD}")
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict(valid_df)
    predictions = [round(value) for value in preds]
    print("f1_Score:",metrics.f1_score(yvalid, preds,average='weighted'))
    joblib.dump(clf, f"models/{MODEL}_breed_{FOLD}.pkl")