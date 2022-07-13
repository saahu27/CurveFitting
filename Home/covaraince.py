from re import U
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import pandas as pd
import numpy as np

input = pd.read_csv(r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Getting started - ENPM673 Spring 2021\data.csv")

x1 = input['age']
y1 = input['charges']

xb1 = np.mean(x1)
yb1 = np.mean(y1)

def normalize(x):
    min = np.min(x)
    max = np.max(x)
    range = max - min

    return [(a - min) / range for a in x],max

x = input['age']
y = input['charges']
x,maxx = normalize(x)
y,maxy = normalize(y)


xb = np.mean(x)
yb = np.mean(y)



def variance(x,xbar):
    sum = 0
    for i in range(0,len(x)):
        sum = sum + (x[i] * x[i])
    x2 = sum/len(x)
    variancex = x2 - (xbar * xbar)
    return variancex

def covariance(x,y,xbar,ybar):
    sum = 0
    for i in range(0,len(x)):
        sum = sum + (x[i] * y[i])
    x2 = sum/len(x)
    cov = x2 - (xbar * ybar)
    return cov


xbar = xb
variancex = variance(x,xbar)

ybar = yb
variancey = variance(y,ybar)
covariancexy = covariance(x,y,xbar,ybar)

covariance_matrix = np.array([[variancex,covariancexy],[covariancexy,variancey]])

print(np.cov(x,y),covariance_matrix)
eigenvalues,eigenvectors = np.linalg.eig(covariance_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]

C = eigenvectors[:,idx]
eigenvectors = C


# print(eigenvalues)
# # print(eigenvectors)
# print(np.var(x),variancex,np.var(y),variancey)

plt.scatter(x1,y1)

# origin = np.array([[xbar, ybar], [xbar, ybar]]).T

# origin = [xbar,ybar]
origin = [xb1,yb1]
plt.quiver(*origin, *eigenvectors[:,0],color=['g'],scale = 10)
plt.quiver(*origin,*eigenvectors[:,1],color=['r'],scale = 10)

# plt.quiver(*origin, eigenvectors[:, 0], eigenvectors[:, 1], color=['black', 'red'],scale =10)
# plt.yticks(np.arange(10000,60000,step=10000))
plt.show()