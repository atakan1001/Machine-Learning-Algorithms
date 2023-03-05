#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 23:03:46 2022

@author: atakanarslan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))

all_data = np.genfromtxt("hw04_data_set.csv",delimiter = ",", skip_header = 1)
training = all_data[:150]
test = all_data[150:]
x_train = training[:,0]
y_train = training[:,1]
x_test = test[:,0]
y_test = test[:,1]

N_train = x_train.shape[0]
N_test = x_test.shape[0]

def minn(arr):
    return np.min(arr)

def sortt(arr):
    return np.sort(arr)

def min_index(arr, minn):
    index = 0
    for i in range(len(arr)):
        if arr[i] == minn:
            index = i
            break
            
    return index

def RMSE(ar, predicted_ar):
    rmse = np.sqrt(np.sum(np.square(ar - predicted_ar))/len(ar))
    return rmse


P = 25

def regression_tree(P):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_frequencies = {}
    
    length = range(N_train)
    
  
    
    node_indices[1] = np.array(length)
    is_terminal[1] = False
    need_split[1] = True
    cont = 1
    split_nodes = []
   
    
    while cont:
        
        split_nodes = [key for key, value in need_split.items()
                   if value == True]
        
                
        if len(split_nodes) == 0:
            break
  
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            if len(data_indices) <= P:
                is_terminal[split_node] = True
                node_frequencies[split_node] = np.sum(y_train[data_indices])/(len(y_train[data_indices]))
            
            else:
                is_terminal[split_node] = False

                best_score = 0.0
                best_split = 0.0
            
                unique_values = sortt(np.unique(x_train[data_indices]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2
                split_scores = np.zeros((split_positions).shape[0])
                for pos in range((split_positions).shape[0]):
                    left_indices = data_indices[x_train[data_indices] > split_positions[pos]]
                    
                    right_indices = data_indices[x_train[data_indices] <= split_positions[pos]]
                
                    split_scores[pos] = (np.sum(np.square(y_train[left_indices] - np.sum(y_train[left_indices])/len(y_train[left_indices]))) + 
                                   np.sum(np.square(y_train[right_indices] - np.sum(y_train[right_indices])/len(y_train[right_indices]))))/ len(data_indices)
                
                
                
                best_score = minn(split_scores)
                best_split = split_positions[min_index(split_scores,best_score)]
            
                node_features[split_node] = best_score
                node_splits[split_node] = best_split
            
             
                left_indices = data_indices[x_train[data_indices] > best_split]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
      
                right_indices = data_indices[x_train[data_indices] <= best_split]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
                
                
    y_predicted_train = np.zeros(N_train)
    
    for i in range(N_train):
        index = 1
        cont = 1
        while cont:
            if is_terminal[index] != False:
                y_predicted_train[i] = node_frequencies[index]
                break
            elif x_train[i] > node_splits[index]:
                index = index * 2
            else:
                index = index * 2 + 1
                    
    err_train = RMSE(y_train, y_predicted_train)  
               
    y_predicted_test = np.zeros(N_test)
    for i in range(N_test):
        index = 1
        cont = 1
        while cont:
            if is_terminal[index] != False:
                y_predicted_test[i] = node_frequencies[index]
                break
            elif x_test[i] > node_splits[index]:
                index = index * 2
            else:
                index = index * 2 + 1
                    
  
    err_test = RMSE(y_test, y_predicted_test)
    
    return node_indices,is_terminal,node_frequencies,node_splits,err_train,err_test
    

node_indices,is_terminal,node_frequencies,node_splits,train_error,test_error = regression_tree(P)

plt.figure(figsize = (10,6))

train_dic = {}
for key,value in node_indices.items():
    for v in value:
        if ((key in node_frequencies) and (x_train[v] not in train_dic)): train_dic[x_train[v]] = node_frequencies[key]

train_dic = {keyy: valuee for keyy, valuee in sorted(train_dic.items(), key=lambda item: item[0])}

plt.plot(x_train,y_train,"b.",markersize = 15)
plt.plot(x_test,y_test,"r.",markersize = 15)
plt.plot(train_dic.keys(),train_dic.values(),"k-")

plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training","test"],loc = 2)
plt.show()

print("RMSE on training set is {} when P is {}".format(str(train_error),str(P)))
print("RMSE on test set is {} when P is {}".format(str(test_error),str(P)))


train_error = np.zeros(10)
test_error = np.zeros(10)
indexx = np.zeros(10)
p = 5
ind = 0
while p <= 50:
    indexx[ind] = p
    node,terminal,freq,splits,train,test = regression_tree(p)
    train_error[ind] = train
    test_error[ind] = test
    p = p + 5
    ind = ind + 1
       
plt.figure(figsize = (10,6))
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.plot(indexx,train_error, ".b-" ,markersize = 15)
plt.plot(indexx,test_error,".r-" ,markersize = 15)
plt.legend(["training" , "test"],loc = 1)
plt.show()
