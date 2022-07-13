from re import U
from cv2 import threshold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import pandas as pd
from sklearn import preprocessing

input = pd.read_csv(r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Getting started - ENPM673 Spring 2021\data.csv")


x = input['age']
y = input['charges']

class linearleastsquare:

    def fit(self,x,y):
        O = np.ones(325)
        x = np.column_stack((O,x))
        y = np.row_stack(y)
        print(y.shape)
        X_dagger = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
        weights = np.dot(X_dagger, y)
        print(weights)
        return weights

O = np.ones(325)
x1 = np.column_stack((O,x))
lls = linearleastsquare()
weights = lls.fit(x,y)
lls_model = x1.dot(weights)

#create basic scatterplot
plt.plot(x, y, 'o')

#add linear regression line to scatterplot 
plt.plot(x,lls_model,color = 'red')
plt.show()