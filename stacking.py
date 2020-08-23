import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier

breed_TrainingData = 'input/breed_train_folds.csv'
pet_TrainingData = 'input/pet_train_folds.csv'


df = pd.read_csv(breed_TrainingData)

def get_stacking():
    level0 = list()
    level0.append(('randomforest', RandomForestClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('xgb', XGBClassifier()))
    #level0.append(('mlp', MLPClassifier()))
    level1 = LogisticRegression(max_iter=3000) # define meta learner model
    model = StackingClassifier(estimators=level0, 
                                final_estimator=level1, cv=5) # define the stacking ensemble
    return model

def get_models():
    models = dict()
    models['stacking'] = get_stacking()
    models['lr'] = LogisticRegression(max_iter=2000)
    models['cart'] = DecisionTreeClassifier(max_depth=10)
    models['randomforest'] = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    models['mlp'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1)
    models['xgb'] = XGBClassifier()
    models['svm'] = SVC(decision_function_shape="ovo")
    return models

def evaluate_model(model, X, y):
    scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1, error_score='raise', verbose=3)
    return scores
 
if __name__ == "__main__":
    X = df.drop(['breed_category','kfold'], axis=1)
    y = ytrain = df.breed_category.values
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
	    scores = evaluate_model(model, X, y)
	    results.append(scores)
	    names.append(name)
	    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()