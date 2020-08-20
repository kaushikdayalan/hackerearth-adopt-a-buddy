import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import numpy as np
import dispatcher
import joblib

MODEL_BREED="xgb_breed"
MODEL_PET = "xgb_pet"
TrainingData = 'input/train_folds.csv'
TestData = 'input/clean_test.csv'

def predict():
    df = pd.read_csv(TestData)
    train_df = pd.read_csv(TrainingData)

    breed_predictions = None
    pet_predictions = None
    pet_id = df['pet_id'].values
    df = df.drop(['pet_id','length(m)','height(cm)'],axis=1)
    for FOLD in range(10):
        print("FOLD:",FOLD)
        clf_breed = joblib.load(os.path.join(f"models/{MODEL_BREED}_breed_{FOLD}.pkl"))
        clf_pet = joblib.load(os.path.join(f"models/{MODEL_PET}_pet_{FOLD}.pkl"))
        breed_preds = clf_breed.predict(df)
        pet_preds = clf_pet.predict(df)
        if FOLD == 0:
            breed_predictions = breed_preds
            pet_predictions = pet_preds
        else:
            breed_predictions += breed_preds
            pet_predictions += pet_preds

    breed_predictions = np.round(breed_predictions/10)  
    pet_predictions = np.round(pet_predictions/10)

    sub = pd.DataFrame(np.column_stack((pet_id, breed_predictions, pet_predictions)),
                        columns=["pet_id","breed_category","pet_category"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"submissions/xgb_tuned_7.csv",index=False) 