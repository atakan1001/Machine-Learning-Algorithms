#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 17:26:04 2022

@author: atakanarslan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

def gradient_W(X, y_truth, y_predicted):
    return(np.asarray([-np.matmul((y_truth[:,c] - y_predicted[:,c])*y_predicted[:,c]*(1-y_predicted[:,c]),X)
                       for c in range(K)]).transpose())

def gradient_w0(y_truth, y_predicted):
    return(np.asarray([-np.sum((y_truth[:,c] - y_predicted[:,c])*y_predicted[:,c]*(1-y_predicted[:,c]) )
                       for c in range(K)]).transpose())

data_set = np.genfromtxt("hw02_data_points.csv", delimiter = ",")

training_set = data_set[:10000]
test_set = data_set[10000:]


N = data_set.shape[0]
NN = training_set.shape[0]
D = data_set.shape[1]

#N = 15000 , D = 784




y_truth = np.genfromtxt("hw02_class_labels.csv",  delimiter = ",").astype(int)
#y_truth is classes

training_class = y_truth[:10000]

test_class = y_truth[10000:]


# K = number of classes
K = np.max(y_truth).astype(int)




N_training = training_class.shape[0]
K_training = np.max(training_class).astype(int)

N_test = test_class.shape[0]
K_test = np.max(test_class).astype(int)

Y_trainingtruth = np.zeros((N_training, K_training)).astype(int)
Y_testtruth = np.zeros((N_test, K_test)).astype(int)



for i in range(N_training):
    Y_trainingtruth[i][training_class[i]-1] = 1 
    
for i in range(N_test):
    Y_testtruth[i][test_class[i]-1] = 1






W = np.genfromtxt("hw02_W_initial.csv", delimiter = ",")
w0 = np.genfromtxt("hw02_w0_initial.csv", delimiter = ",")

eta = 0.00001
iteration = 1
iteration_count = 1000
objective_values = []
 



while iteration <= iteration_count:
    Y_predicted = sigmoid(training_set, W, w0)

    objective_values = np.append(objective_values, 0.5*(np.sum((Y_trainingtruth - Y_predicted)**2)))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(training_set, Y_trainingtruth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_trainingtruth, Y_predicted)

    iteration = iteration + 1

print("W = \n",W)
print("w0 = \n",w0)



plt.figure(figsize = (20, 10))
plt.plot(range(1, iteration ), objective_values, "b-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_train = pd.crosstab(y_predicted, training_class.transpose(),
                               rownames = ["Y_Prediction"],
                               colnames = ["Y_Truth"])

print("\n \n")
print(confusion_train)

Y_testpredicted = sigmoid(test_set,W,w0)
y_testpredicted = np.argmax(Y_testpredicted, axis =1) + 1
confusion_test = pd.crosstab(y_testpredicted, test_class.transpose(),
                               rownames = ["Y_Prediction"],
                               colnames = ["Y_Truth"])
       
print("\n \n")
print(confusion_test)  
    
    
    
