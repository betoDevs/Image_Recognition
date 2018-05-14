# -*- coding: utf-8 -*-
"""
@author: Beto

"""
from functions import load_data, prepare_data_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 
    
wdir = "waldo_data/"

waldos_train = load_data(wdir + "waldo/")
notwaldos_train = load_data(wdir + "notwaldo/")
waldos_test = load_data(wdir + "waldo-test/")
notwaldos_test = load_data(wdir + "notwaldo-test/")
allwaldos_train, y = prepare_data_classification(
        waldos_train, notwaldos_train)
rf = RandomForestClassifier(bootstrap=True, n_estimators=60, 
                            max_features='sqrt', oob_score=True)
rf = rf.fit(allwaldos_train,y)
print(f'\nOut-of-bag score estimate: {rf.oob_score_:.3}')

filename = "model_60.sav"
joblib.dump(rf, filename)

allwaldos_test, y_test = prepare_data_classification(
        waldos_test, notwaldos_test)
predicted = rf.predict(allwaldos_test)
print("\nWaldos Prediction")
print(predicted)
print("\nActual Waldos")
print(y_test)
accuracy = accuracy_score(y_test, predicted)
print(f'\nOut-of-bag score estimate: {rf.oob_score_:.3}')