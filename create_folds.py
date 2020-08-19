import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('input/clean_train.csv')
    df['kfold'] = -1
    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=df.breed_category.values)):
        print(len(train_idx), len(valid_idx))
        df.loc[valid_idx, 'kfold'] = fold
    print(df.head())
    df.to_csv('input/train_folds.csv',index=False)
