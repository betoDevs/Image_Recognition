# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:32:46 2018

@author: 0888292
"""
from functions import load_data, blurr
from sklearn.externals import joblib 

wdir = "waldo_data/full_picture/"
savespot = "waldo_data/tested_picture/picture.jpeg"
loadspot = "waldo_data/tested_picture/"
picture = load_data(wdir)
picture_copy = picture[0]

model_1 = joblib.load("model_60.sav")

rows = len(picture[0,]) // 64
columns = (len(picture[0,1]) // 64) 

tresh = 0.70
for i in range(0, rows):
    for j in range(0, columns): 
        slice_piece = picture_copy[64*i:64+64*i, 64*j:64+64*j]
        data = slice_piece.flatten('F')
        data = data.reshape(1,-1)
        prediction = model_1.predict_proba(data)
        if prediction[0][1] < tresh:
            blurr(picture[0], 64*j,64*i,64+64*j,64+64*i,
                  savespot, 'JPEG')
            picture = load_data(loadspot) 
