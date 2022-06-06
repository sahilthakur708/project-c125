#Importing all the important models and install them if not installed on your device
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nClasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=9,train_size=3500,test_size=2500)
xtrain_scaled=xtrain/255.0
xtest_scaled=xtest/255.0

clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrain_scaled,ytrain)
ypred=clf.predict(xtest_scaled)
accuracy=accuracy_score(ytest,ypred)
print(accuracy)

def getPrediction(image):
  # convert to grayscale image - 'L' format means each pixel is 
  # represented by a single value from 0 to 255
  im_pil=Image.open(image)
  image_bw = im_pil.convert('L')
  image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

  image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
  pixel_filter = 20
  min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
  image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
  max_pixel = np.max(image_bw_resized_inverted)
  image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
  test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
  test_pred = clf.predict(test_sample)
  return test_pred[0]
