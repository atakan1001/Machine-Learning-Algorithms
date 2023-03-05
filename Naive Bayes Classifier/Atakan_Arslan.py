#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:50:23 2022

@author: atakanarslan
"""

import numpy as np

filename = "hw01_data_points.csv"
filename1 = "hw01_class_labels.csv"

data = np.loadtxt(filename,delimiter = ",",dtype = "str")
label = np.loadtxt(filename1,dtype = "int")

# np.savetxt("output.csv",data,delimiter = ",",fmt = "%s")

training_set = (data[0:300])
test_set = data[300:400]
training_label = label[:300]
test_label = label[300:]

pAcd = np.array([[0.0]*7, [0.0]*7])
pCcd = np.array([[0.0]*7, [0.0]*7])
pGcd = np.array([[0.0]*7, [0.0]*7])
pTcd = np.array([[0.0]*7, [0.0]*7])

labellen = [ len(training_label[training_label == 1]),len(training_label[training_label == 2])]
 

for idx,arr in enumerate(training_set):
    theLabel = training_label[idx]
    for i in range(7):
        base = arr[i]
       
        if base == "A": pAcd[theLabel-1][i] += 1/int(labellen[theLabel-1])
        elif base == "C": pCcd[theLabel-1][i] += 1/labellen[theLabel-1]
        elif base == "G": pGcd[theLabel-1][i] += 1/labellen[theLabel-1]
        elif base == "T": pTcd[theLabel-1][i] += 1/labellen[theLabel-1]
     
        
     
class_priors = [labellen[0]/len(training_label),labellen[1]/len(training_label)]


confusion_train = [[0,0],[0,0]]


for ind,series in enumerate(training_set):
    labell = training_label[ind]
    carp1 = 0
    carp2 = 0
    for j in range(7):
        ch = series[j]
        
        if ch == "A" :
            carp1 += np.log(pAcd[0][j])
            carp2 += np.log(pAcd[1][j])
        elif ch == "C" :
            carp1 += np.log(pCcd[0][j])
            carp2 += np.log(pCcd[1][j])  
        elif ch == "G" :
            carp1 += np.log(pGcd[0][j])
            carp2 += np.log(pGcd[1][j])   
        elif ch == "T" :
            carp1 += np.log(pTcd[0][j])
            carp2 += np.log(pTcd[1][j])  
    
    if (labell == 1 and (carp1 > carp2)) :
            confusion_train[0][0] += 1
        
    elif (labell == 1 and (carp1 < carp2)) :
            confusion_train[1][0] += 1
            
    elif (labell == 2 and (carp1 < carp2)) :
            confusion_train[1][1] += 1  
            
    elif (labell == 2 and (carp1 > carp2)) :
            confusion_train[0][1] += 1
            
print("Confusion Train")
print("y_truth  1     2 \ny_pred\n1      ",confusion_train[0][0], " " ,confusion_train[0][1], "\n2       ",confusion_train[1][0], "  " ,confusion_train[1][1],"\n")      


confusion_test = [[0,0],[0,0]]


for ind,series in enumerate(test_set):
    labell = test_label[ind]
    carp1 = 1
    carp2 = 1
    for j in range(7):
        ch = series[j]
        
        if ch == "A" :
            carp1 += np.log(pAcd[0][j])
            carp2 += np.log(pAcd[1][j])
        elif ch == "C" :
            carp1 += np.log(pCcd[0][j])
            carp2 += np.log(pCcd[1][j])  
        elif ch == "G" :
            carp1 += np.log(pGcd[0][j])
            carp2 += np.log(pGcd[1][j])   
        elif ch == "T" :
            carp1 += np.log(pTcd[0][j])
            carp2 += np.log(pTcd[1][j])  
    
    if (labell == 1 and (carp1 > carp2)) :
            confusion_test[0][0] += 1
        
    elif (labell == 1 and (carp1 < carp2)) :
            confusion_test[1][0] += 1
            
    elif (labell == 2 and (carp1 < carp2)) :
            confusion_test[1][1] += 1  
            
    elif (labell == 2 and (carp1 > carp2)) :
            confusion_test[0][1] += 1         
            
print("Confusion Test")
print("y_truth  1     2 \ny_pred\n1       ",confusion_test[0][0], "  " ,confusion_test[0][1], "\n2        ",confusion_test[1][0], " " ,confusion_test[1][1])   


