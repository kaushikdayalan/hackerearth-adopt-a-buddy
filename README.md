# hackerearth-adopt-a-buddy

This was my solution for the ML competition held my Hackerearth.
It achieved a score of 90.81 on the final LB.

![result](/images/upload.png)

The scripts are a bit sketchy I will fix them.

## Notebook 
The notebook contains feature engineering and creating the cleaned datasets.

## create_folds.py
This file implements stratified kfold to the dataset.
The dataset had an imbalance in the data so I have implemented smote to over sample the minority classes

## dispatcher.py
This file contains the classifiers used for training

## train.py
This file trains the models. When saving model is set to true. Make sure you have a `/models` folder in the current directory

## stacking.py
This file is an implementation of stacking model ensemble using sklearn. 
I didnt have the compute power to tune this model with the given time so I just used the baseline
model to get a score. The score was higher than all the classifiers used but I didnt want to tune the model.
```
sklearn.ensemble.StackingClassifier
```

## predict.py
This file is used to make predictions using the trained models and create a submission file.
Please note that if you are trying to recreate and get a submission file then you need to create a `/submissions` folder in the current path

## Best result
Although I tried many models, XGBoostClassifier gave the highest score after hyperparameter tuning.
(hyperparameter tuning is not included in the repo)
