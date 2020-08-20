import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import dispatcher
import joblib


BREED_MODEL = "xgb"
PET_MODEL = "xgb"
category = 'both'
save = True
breed_TrainingData = 'input/breed_train_folds.csv'
pet_TrainingData = 'input/pet_train_folds.csv'
FOLD_MAPPING = {
    0:[1, 2, 3, 4, 5, 6, 7, 8, 9],
    1:[0, 2, 3, 4, 5, 6, 7, 8, 9],
    2:[0, 1, 3, 4, 5, 6, 7, 8, 9],
    3:[0, 1, 2, 4, 5, 6, 7, 8, 9],
    4:[0, 1, 2, 3, 5, 6, 7, 8, 9],
    5:[0, 1, 2, 3, 4, 6, 7, 8, 9],
    6:[0, 1, 2, 3, 4, 5, 7, 8, 9],
    7:[0, 1, 2, 3, 4, 5, 6, 8, 9],
    8:[0, 1, 2, 3, 4, 4, 6, 7, 9],
    9:[0, 1, 2, 3, 4, 5, 6, 7, 8]
}

if __name__ == "__main__":
    breed_df = pd.read_csv(breed_TrainingData)
    pet_df = pd.read_csv(pet_TrainingData)
    bred_predictions = None
    pet_predictions = None
    if category in ["breed","both"]: 
        for i in range(10):
            FOLD = i
            
            train_df = breed_df[breed_df.kfold.isin(FOLD_MAPPING.get(FOLD))]
            valid_df = breed_df[breed_df.kfold == FOLD]
            ytrain = train_df.breed_category.values
            yvalid = valid_df.breed_category.values
           
           
            train_df = train_df.drop(['breed_category','kfold'], axis=1)
            valid_df = valid_df.drop(['breed_category','kfold'], axis=1)
            valid_df = valid_df[train_df.columns]
            print(f"Model: {BREED_MODEL}, FOLD: {FOLD}")
            print()
            clf = dispatcher.MODELS[BREED_MODEL]
            clf.fit(train_df, ytrain)
            train_preds = clf.predict(train_df)
            preds = clf.predict(valid_df)
            print("breed train f1_Score:",metrics.f1_score(ytrain, train_preds,average='weighted'))
            print("breed valid f1_Score:",metrics.f1_score(yvalid, preds,average='weighted'))
            #print(metrics.classification_report(yvalid, preds))
            #print(pd.crosstab(yvalid, preds))
            if save == True:
                joblib.dump(clf, f"models/{BREED_MODEL}_breed_{FOLD}.pkl")
            print()
        print('-----------------------------------------------------------------------')
    if category in ["pet",'both']:
        for i in range(10):
            FOLD = i
            train_df = pet_df[pet_df.kfold.isin(FOLD_MAPPING.get(FOLD))]
            valid_df = pet_df[pet_df.kfold == FOLD]

            ytrain = train_df.pet_category.values
            yvalid = valid_df.pet_category.values

            train_df = train_df.drop(['pet_category','kfold'], axis=1)
            valid_df = valid_df.drop(['pet_category','kfold'], axis=1)

            valid_df = valid_df[train_df.columns]
            print(f"Model: {PET_MODEL}, FOLD: {FOLD}")
            clf = dispatcher.MODELS[PET_MODEL]
            clf.fit(train_df, ytrain)
            train_preds = clf.predict(train_df)
            preds = clf.predict(valid_df)
            print("pet train f1_Score:",metrics.f1_score(ytrain, train_preds,average='weighted'))
            print("pet valid f1_Score:",metrics.f1_score(yvalid, preds,average='weighted'))
            #print(metrics.classification_report(yvalid, preds))
            #print(pd.crosstab(yvalid, preds))
            if save == True:
                joblib.dump(clf, f"models/{PET_MODEL}_pet_{FOLD}.pkl")