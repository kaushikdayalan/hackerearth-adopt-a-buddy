import pandas as pd
from sklearn import model_selection

def kfold_create(category):
    df = pd.read_csv('input/clean_train.csv')
    df['kfold'] = -1
    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    if category == 'breed_category':
        print("Creating breed category training fold")
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=df.breed_category.values)):
            print(len(train_idx), len(valid_idx))
            df.loc[valid_idx, 'kfold'] = fold
        print(df.head())
        df.to_csv('input/breed_train_folds.csv',index=False)

    elif category == 'pet_category':
        print("Creating pet category training fold")
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=df.pet_category.values)):
            print(len(train_idx), len(valid_idx))
            df.loc[valid_idx, 'kfold'] = fold
        print(df.head())
        df.to_csv('input/pet_train_folds.csv',index=False)

if __name__ == "__main__":
    kfold_create(category="pet_category")

