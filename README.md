# hackerearth-adopt-a-buddy

This was my solution for the ML competition held my Hackerearth.
It achieved a score of 90.91.

![result](/images/upload.png)

## Notebook 
The notebook contains feature engineering and creating the cleaned datasets.

## create_folds.py
This file implements stratified kfold to the dataset.
The dataset had an imbalance in the data so I have implemented smote to over sample the minority classes

## dispatcher.py
This file contains the classifiers used for training

## train.py
This file trains the models.

## stacking.py
This file is an implementation of stacking model ensemble using sklearn. 
I didnt have the compute power to tune this model with the given time so I just used the baseline
model to get a score. The score was higher than all the classifiers used but I didnt want to tune the model.
`<addr>sklearn.ensemble.StackingClassifier`

## predict.py
This file is used to make predictions using the trained models and create a submission file.

## Best result
Although I tried many models, XGBoostClassifier gave the highest score after hyperparameter tuning.
(hyperparameter tuning is not included in the repo)
