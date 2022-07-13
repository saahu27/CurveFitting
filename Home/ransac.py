from re import U
from cv2 import threshold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
import pandas as pd
from sklearn import preprocessing
import math

input = pd.read_csv(r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Getting started - ENPM673 Spring 2021\data.csv")


x = input['age']
y = input['charges']
x1 = x
y1 = y
O = np.ones(325)
x = np.column_stack((O,x))

class linearleastsquare:

    def fit(self,x,y):
        X_dagger = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
        weights = np.dot(X_dagger, y)
        return weights

class Ransac:
    def __init__(self, weights):
        self.weights = weights
    
    def fit(self,x,y,threshold):

        num_iter = math.inf
        num_sample = 2

        max_inlier_count = 0
        best_model = None

        desired_prob = 0.95
        prob_outlier = 0.5
        
        data = np.column_stack((x, y)) 
        data_size = len(data)

        iter_done = 0

        while num_iter > iter_done:

            np.random.shuffle(data)
            sample_data = data[:num_sample, :]
            estimated_model = self.weights.fit(sample_data[:,:-1], sample_data[:, -1])

            y_cap = x.dot(estimated_model)
            err = np.abs(y - y_cap.T)
            inlier_count = np.count_nonzero(err < threshold)
 
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model


            prob_outlier = 1 - inlier_count/data_size
            print('No.of inliers:', inlier_count)
            print('probability of outlier:', prob_outlier)
            num_iter = np.ceil(math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**num_sample))
            iter_done = iter_done + 1

            print('Total No of iterations done:', iter_done)
            print('no.of.iterations expected:', num_iter)
            print('max_inlier_count: ', max_inlier_count)

        return best_model

lls = linearleastsquare()
ransac_model = Ransac(lls)
threshold = np.std(y)/2
est = ransac_model.fit(x,y,threshold)
ransac = x.dot(est)

plt.plot(x1,y1,'o',color = (0,1,0))
plt.plot(x1,ransac,color ='red')
plt.show()

