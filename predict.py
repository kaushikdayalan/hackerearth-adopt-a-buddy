import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import numpy as np
import dispatcher
import joblib

MODEL_BREED="randomforest"
MODEL_PET = "randomforest2"
TrainingData = 'input/train_folds.csv'
TestData = 'input/clean_test.csv'

def predict():
    df = pd.read_csv(TestData)
    train_df = pd.read_csv(TrainingData)
    pet_df = df.copy()
    pet_df['breed_category'] = train_df['breed_category']
    breed_predictions = None
    pet_predictions = None
    pet_id = pet_df['pet_id'].values
    pet_df = pet_df.drop(['pet_id'],axis=1)
    df = df.drop(['pet_id'],axis=1)
    for FOLD in range(5):
        print("FOLD:",FOLD)
        clf_breed = joblib.load(os.path.join(f"models/{MODEL_BREED}_breed_{FOLD}.pkl"))
        clf_pet = joblib.load(os.path.join(f"models/{MODEL_PET}_pet_{FOLD}.pkl"))
        breed_preds = clf_breed.predict(df)
        pet_preds = clf_pet.predict(pet_df)
        if FOLD == 0:
            breed_predictions = breed_preds
            pet_predictions = pet_preds
        else:
            breed_predictions += breed_preds
            pet_predictions += pet_preds

    breed_predictions = np.round(breed_predictions/5)  
    pet_predictions = np.round(pet_predictions/5)

    sub = pd.DataFrame(np.column_stack((pet_id, breed_predictions, pet_predictions)),
                        columns=["pet_id","breed_category","pet_category"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/combi.csv",index=False) 