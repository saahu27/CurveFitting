from re import U
from cv2 import threshold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import pandas as pd


x1 = 5
y1 = 5
xp1 = 100
yp1 = 100
x2 = 150
y2 = 5
xp2 = 200
yp2 = 80
x3 = 150
y3 = 150
xp3 = 220
yp3 = 80
x4 = 5
y4 = 150
xp4 = 100
yp4 = 200

r1 = [-x1,-y1,-1,0,0,0,x1 *xp1,y1 * xp1,xp1]
r2 = [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1]
r3 = [-x2,-y2,-1,0,0,0,x2 *xp2,y2 * xp2,xp2]
r4 = [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2]
r5 = [-x3,-y3,-1,0,0,0,x3*xp3,y3 * xp3,xp3]
r6 = [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3]
r7 = [-x4,-y4,-1,0,0,0,x4*xp4,y4 * xp4,xp4]
r8 = [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]

# x = [h11,h12,h13,h21,h22,h23,h31,h32,h33]

A = np.row_stack((r1,r2,r3,r4,r5,r6,r7,r8))
# print(A.shape)


A2 = np.matmul(A,A.T)

lefteigenvalues,lefteigenvectors = np.linalg.eig(A2)
lefteigenvalues = np.abs(lefteigenvalues)

idx = lefteigenvalues.argsort()[::-1]
lefteigenvalues = lefteigenvalues[idx]

idx = lefteigenvalues.argsort()[::-1]  
U = lefteigenvectors[:,idx]
lefteigenvectors = U

print("U",U)

leftsingularvalues = []
for i in range(0,len(lefteigenvalues)):
    if lefteigenvalues[i]!=0:
        leftsingularvalues.append(np.sqrt(np.abs(lefteigenvalues[i])))

leftsingularvalues = np.sort(leftsingularvalues)
leftsingularvalues = np.flipud(leftsingularvalues)

print("singularvalues",leftsingularvalues)

D=np.diag(leftsingularvalues)
temp = np.array([[0]]*8)
D = np.append(D, temp, axis = 1)
print(D)

A3 = np.matmul(A.T,A)

righteigenvalues,righteigenvectors = np.linalg.eig(A3)
righteigenvalues = np.abs(righteigenvalues)
righteigenvalues = righteigenvalues[righteigenvalues.argsort()[::-1]]

idx = righteigenvalues.argsort()[::-1]  
V = righteigenvectors[:,idx]

print("V",V)

H = V[:,8]
H = np.reshape(H,(3,3))
print(H)

normalizer = H[2,2]
print(normalizer)

for i in range(0,3):
    for j in range(0,3):
        H[i,j] = H[i,j]/normalizer
print(H)