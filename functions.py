# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:06:07 2018

@author: 0888292
"""
import numpy as np
from matplotlib.pyplot import imread
import pandas as pd
import os
import PIL
from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageEnhance

#Loads data for specified path.
def load_data(datapath):
    waldos_train = np.array([np.array(imread(datapath + fname)) 
        for fname in os.listdir(datapath)])
    
    return waldos_train

#params: data for classification purposes
#returns: Two arrays: input vector x & output label vector y 
def prepare_data_classification(
        waldos_train, notwaldos_train):
    data = []
    y = []
    for im in waldos_train:
        data.append(im.flatten('F'))
        y.append(1)
    df1 = pd.DataFrame(data)
    df_y1 = pd.DataFrame(y)
    
    data = []
    y = []
    for im in notwaldos_train:
        data.append(im.flatten('F'))
        y.append(0)
    df2 = pd.DataFrame(data)
    df_y2 = pd.DataFrame(y)
    
    frames_x = [df1, df2]
    frames_y = [df_y1, df_y2]
    
    allwaldos_train_x = pd.concat(frames_x)
    allwaldos_train_y = pd.concat(frames_y)
    
    return allwaldos_train_x, allwaldos_train_y

def prepare_data_classification2(raw_data):
    
    data = []
    for im in raw_data:
        data.append(im.flatten('F'))
    prepared_data = pd.DataFrame(data)
    return prepared_data

#params: A picture to be diced by a 64x64 with an overlap of 50%
#returns: Diced data ready to be tested.
def slice_picture(pic):
    rows = len(pic[0,]) // 64
    columns = (len(pic[0,1]) // 64) - 1
    
    row_overlap = rows + int(rows*0.50)
    column_overlap = columns + int(columns*0.50)
#    print(rows)
#    print(columns)
#    print(row_overlap)
#    print(column_overlap)
    data = []
    for i in range(0, row_overlap):
        for j in range(0, column_overlap):
            slice_piece = pic[0, 32*i:64+32*i, 32*j:64+32*j]
            data.append(slice_piece.flatten('F'))       
    
    data = pd.DataFrame(data)
    return data

def blurr(picture, top_left_x, top_left_y,
          bottom_right_x, bottom_right_y):
    picture = Image.fromarray(picture, 'RGB')
#    draw = Image.Draw(picture)   
    cropped_image = picture.crop((top_left_x, top_left_y,
                                  bottom_right_x, bottom_right_y))
    blurred_image = cropped_image.convert('L')
    picture.paste(blurred_image,(top_left_x, top_left_y,
                                 bottom_right_x, bottom_right_y))
    picture.save("Hey-Waldo/full_fake_picture/pic_copy.jpeg", 'JPEG')
    
    
    
    