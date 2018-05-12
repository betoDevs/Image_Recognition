# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:32:46 2018

@author: 0888292
"""
from functions import *
import PIL
from PIL import Image
import pandas as pd
from sklearn.externals import joblib 

wdir = "Hey-Waldo/full_picture/"
savespot = "Hey-Waldo/sliced_picture/"
name = "sliced_pic_"
picture = load_data(wdir)
picture_copy = picture[0]

model_1 = joblib.load("model_60.sav")

rows = len(picture[0,]) // 64
columns = (len(picture[0,1]) // 64) 

row_overlap = rows# + int(rows*0.50)
column_overlap = columns# + int(columns*0.50)

print(rows)
print(columns)
print(row_overlap)
print(column_overlap)

#slice_sliding
#picture_copy[32*i:64+32*i, 32*j:64+32*j]
#blurr(picture[0], 32*j,32*i,64+32*j,64+32*i)

tresh = 0.70
for i in range(0, row_overlap):
    for j in range(0, column_overlap): 
        slice_piece = picture_copy[64*i:64+64*i, 64*j:64+64*j]
        #img = Image.fromarray(slice_piece, 'RGB')
        data = slice_piece.flatten('F')
        data = data.reshape(1,-1)
        prediction = model_1.predict_proba(data)
        if prediction[0][1] < tresh:
            blurr(picture[0], 64*j,64*i,64+64*j,64+64*i)
            picture = load_data("Hey-Waldo/full_fake_picture/") 
        #new_name = savespot + name + str(i) + "_" + str(j) + ".jpeg"
        #img.save(new_name,'JPEG')

#picture_copy = Image.fromarray(picture_copy, 'RGB')
#picture_copy.save(wdir+"pic_copy.jpeg", 'JPEG')


#slice picture with a 32 step for a 50% overlap. save it in np array then predict.