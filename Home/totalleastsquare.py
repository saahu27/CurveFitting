
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pandas as pd

input = pd.read_csv(r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Getting started - ENPM673 Spring 2021\data.csv")

x1 = input['age']
y1 = input['charges']

def mean(x):
    sum = 0
    for i in range(0,len(x)):
        sum = sum + x[i]
    return sum/len(x)


def normalize(x):
    min = np.min(x)
    max = np.max(x)
    range = max - min

    return [(a - min) / range for a in x],max

x = input['age']
y = input['charges']
x,maxx = normalize(x)
y,maxy = normalize(y)
x = np.row_stack(x)
y = np.row_stack(y)

class total_least_square:
    def moment_matrix(self,x,xbar):
        u = []
        for i in range(0,len(x)):
            u.append(x[i] - xbar)
        return u

    def second_moment_matrix(self,U) :
        second_moment = np.matmul(U.T,U)
        return second_moment


xbar = mean(x)
ybar = mean(y)

tls = total_least_square()
ux = tls.moment_matrix(x,xbar)
uy = tls.moment_matrix(y,ybar)

U = np.column_stack((ux,uy))
print(U.shape)

second_moment = tls.second_moment_matrix(U)
print(second_moment.shape)
eigenvalues,eigenvectors = np.linalg.eig(second_moment)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]

C = eigenvectors[:,idx]
lefteigenvectors = C
print(eigenvalues,eigenvectors)

N = eigenvectors[:,1]

d = N[0]*xbar + N[1]*ybar


line = (d - (N[0]*x))/N[1]
plt.plot(x, y, 'o')
plt.plot(x,line)
plt.show()
