import pandas as pd
from sklearn import model_selection
from imblearn.over_sampling import SMOTE


def kfold_create(category):
    df = pd.read_csv('input/new_clean_train.csv')

    df['kfold'] = -1
    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    if category == 'breed_category':
        smote = SMOTE()
        x_train = pd.DataFrame()
        x_train = df.drop(['breed_category'],axis=1)
        y_train = df['breed_category']
        x_train, y_train = smote.fit_sample(x_train, y_train)
        print(y_train.value_counts())
        x_train['breed_category'] = y_train
        y_train = y_train.values
        print("Creating breed category training fold")
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=x_train, y=y_train)):
            print(len(train_idx), len(valid_idx))
            x_train.loc[valid_idx, 'kfold'] = fold
        print(x_train.head())
        x_train.to_csv('input/breed_train_folds.csv',index=False)

    elif category == 'pet_category':
        smote = SMOTE()
        x_train = pd.DataFrame()
        x_train = df.drop(['breed_category','pet_category'],axis=1)
        y_train = df['pet_category']
        x_train, y_train = smote.fit_resample(x_train, y_train)
        x_train['pet_category'] = y_train
        print(y_train.value_counts())
        y_train = y_train.values
        print("Creating pet category training fold")
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=x_train, y=y_train)):
            print(len(train_idx), len(valid_idx))
            x_train.loc[valid_idx, 'kfold'] = fold
        print(x_train.head())
        x_train.to_csv('input/pet_train_folds.csv',index=False)

if __name__ == "__main__":
    kfold_create(category="pet_category")

