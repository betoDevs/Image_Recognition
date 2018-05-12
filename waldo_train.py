# -*- coding: utf-8 -*-
"""
@author: Beto

"""
from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 
    
wdir = "Hey-Waldo/"

waldos_train = load_data(wdir + "waldo/")
notwaldos_train = load_data(wdir + "notwaldo/")

img = Image.fromarray(waldos_train[0], 'RGB')
img.show()
waldos_test = load_data(wdir + "waldo-test/")
notwaldos_test = load_data(wdir + "notwaldo-test/")
print(waldos_train.shape)

allwaldos_train, y = prepare_data_classification(
        waldos_train, notwaldos_train)

rf = RandomForestClassifier(n_estimators=60, oob_score=True)
rf = rf.fit(allwaldos_train,y)

filename = "model_60.sav"
joblib.dump(rf, filename)

allwaldos_test, y_test = prepare_data_classification(
        waldos_test, notwaldos_test)
print(allwaldos_test.shape)
predicted = rf.predict_proba(allwaldos_test)
print("\nWaldos Prediction")
print(predicted)
accuracy = accuracy_score(y_test, predicted)
print("\nActual Waldos")
print(y_test)
print(f'\nOut-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')