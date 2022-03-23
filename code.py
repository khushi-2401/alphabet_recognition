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
import os,ssl,time

if(not os.environ.get('PYTHONHTTPSVERIFY','') and
    getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context=ssl._create_unverified_context

X=np.load("image.npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=1,train_size=7500,test_size=2500)
x_train_scaled=x_train/255.0
x_test_scaled=x_test/255.0
clf=LogisticRegression().fit(x_train_scaled,y_train)
y_pred=clf.predict(x_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)