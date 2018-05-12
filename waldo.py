# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:39:32 2018

@author: 0888292
"""

from functions import *
from sklearn.externals import joblib 
import PIL
from PIL import Image

model_1 = joblib.load("model_60.sav")
raw_data = load_data("Hey-Waldo/sliced_picture/")
print(raw_data.shape)

data = prepare_data_classification2(raw_data)
print(data.shape)
prediction = model_1.predict_proba(data)
tresh = 0.80
print(prediction)

position = 0
for i in prediction:
    position+=1
    if i[1] > tresh:
        print("waldo")
        print(position)
    

