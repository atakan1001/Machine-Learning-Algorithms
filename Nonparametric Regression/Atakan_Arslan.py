#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:43:38 2022

@author: atakanarslan
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#importing data
data_set = np.genfromtxt("hw03_data_set.csv",delimiter = ",")

training = data_set[1:151]
test = data_set[151:]

N = data_set.shape[0] - 1  # Number of total data points

x_train = training[:,0] #eruptions in training set
y_train = training[:,1].astype(int) #waitings in training set

x_test = test[:,0]
y_test = test[:,1].astype(int)

bin_width = 0.37
origin = 1.5

min_value = origin
max_value = bin_width * ((max(data_set[1:,0]) - origin) // bin_width + 1) + origin


left_borders = np.arange(start = min_value,
                         stop = max_value,
                         step = bin_width)

right_borders = np.arange(start = min_value + bin_width,
                         stop = max_value + bin_width,
                         step = bin_width)

g_hat = [0] * len(left_borders)

for i in range(len(left_borders)):
    g_hat[i] = y_train[(left_borders[i] < x_train) & (x_train <= right_borders[i])].mean()
    
plt.figure(figsize = (10 , 6))
plt.plot(x_train, y_train, "b.", markersize = 12, label = "training")
plt.plot(x_test, y_test, "r.", markersize = 12, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 2)

border = 0
while border < len(left_borders):
    plt.plot([left_borders[border], right_borders[border]], [g_hat[border], g_hat[border]], "k-")
    border += 1
 
border = 0
while border < len(right_borders) -1:
    plt.plot([right_borders[border], right_borders[border]],[g_hat[border], g_hat[border + 1]], "k-")
    border += 1

plt.show()

# Calculation of RMSE

error = 0
for i in range(len(right_borders)):
    error += sum(((y_test[(left_borders[i] < x_test) & (x_test <= right_borders[i])] - g_hat[i]) **2))

RMSE = np.sqrt(error) / np.sqrt(len(x_test))
print("Regressogram => RMSE is", RMSE, "when h ise", bin_width)

bin_width = 0.37
data_interval = np.linspace(min_value, max_value, 2001)

g_hat = ([np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) * y_train) / np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) 
                    for x in data_interval])

plt.figure(figsize = (10 , 6))
plt.plot(x_train, y_train, "b.", markersize = 12, label = "training")
plt.plot(x_test, y_test, "r.", markersize = 12, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 2)
plt.plot(data_interval, g_hat, "k-")
plt.show()

left_borders = data_interval[:-1]
right_borders = data_interval[1:]

g_hatt = ([np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) * y_train) / np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) 
                      for x in x_test])

RMSE = np.sqrt(np.sum((y_test - g_hatt) **2) / len(y_test))

print("Running Mean Smoother => RMSE is", RMSE, "when h is", bin_width)

g_hat = ([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2) * y_train)/ np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) 
          for x in data_interval])


plt.figure(figsize = (10 , 6))
plt.plot(x_train, y_train, "b.", markersize = 12, label = "training")
plt.plot(x_test, y_test, "r.", markersize = 12, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 2)
plt.plot(data_interval, g_hat, "k-")
plt.show()

g_hatt = ([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2) * y_train)/ np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) 
           for x in x_test])

RMSE = np.sqrt(np.sum((y_test - g_hatt) **2) / len(y_test))

print("Kernel Smoother => RMSE is", RMSE, "when h is", bin_width)