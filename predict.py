import pandas as  pd
from sklearn import ensemble
import os
from sklearn import metrics
import numpy as np
import dispatcher
import joblib

MODEL_BREED="xgb"
MODEL_PET = "xgb"
TestData = 'input/new_clean_test.csv'
test = "input/test.csv"
def predict():
    df = pd.read_csv(TestData)
    testdf = pd.read_csv(test)
    breed_predictions = None
    pet_predictions = None
    pet_id = testdf['pet_id']
    
    for FOLD in range(10):
        print("FOLD:",FOLD)
        clf_breed = joblib.load(os.path.join(f"models/{MODEL_BREED}_breed_{FOLD}.pkl"))
        clf_pet = joblib.load(os.path.join(f"models/{MODEL_PET}_pet_{FOLD}.pkl"))

        pet_preds = clf_pet.predict(df)
        df['pet_category'] = pet_preds
        breed_preds = clf_breed.predict(df)
        df = df.drop(['pet_category'],axis=1)
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
    submission.to_csv(f"submissions/nf_smote_xgb_kfold=10_1.csv",index=False) 