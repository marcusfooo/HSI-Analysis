import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from yahoo import fetch_yahoo

import warnings
from time import sleep


warnings.filterwarnings('ignore')

# reading the data
df = pd.read_csv('HSI.csv')
df.dropna(inplace=True)

# Split dataset
x = df.iloc[:, 2:4]
y = df.iloc[:, 8]
validation_size = 0.20
seed = 7
# Use only training data
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)

def train_accuracy():
# Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('CART', DecisionTreeClassifier()))


    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def predict():
    new_x = fetch_yahoo()
    data_x = pd.DataFrame(data=new_x)
    data_x.set_index('Open', inplace=True)
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    prediction = lr.predict(data_x)

    prediction_str = ''
    if prediction == 1:
        prediction_str = 'RISE'
    else:
        prediction_str = 'FALL'
    print('HSI is expected to %s today!' % prediction_str)

predict()
sleep(3)


