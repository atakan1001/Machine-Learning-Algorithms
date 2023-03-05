#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 19:29:16 2023

@author: atakanarslan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def draw_roc_curve(true_labels, predicted_probabilities):
    tpr = []
    fpr = []
    auc = 0
    sort_index = np.argsort(predicted_probabilities)[::-1]
    true_labels = true_labels[sort_index]
    predicted_probabilities = predicted_probabilities[sort_index]

    for i in range(len(true_labels)):
        tp = sum((predicted_probabilities >= predicted_probabilities[i]) & (true_labels == 1))
        fp = sum((predicted_probabilities >= predicted_probabilities[i]) & (true_labels == -1))
        fn = sum((predicted_probabilities < predicted_probabilities[i]) & (true_labels == 1))
        tn = sum((predicted_probabilities < predicted_probabilities[i]) & (true_labels == -1))
        tpr.append(tp/sum(true_labels==1))
        fpr.append(fp/sum(true_labels==-1))
        if len(fpr) > 1:
            auc += (fpr[-1]-fpr[-2]) * (tpr[-1]+tpr[-2])/2
    plt.plot(fpr, tpr)
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.title('ROC curve')
    plt.show()
    print("The area under the ROC curve is:", auc)




def draw_pr_curve(true_labels, predicted_probabilities):
    thresholds = sorted(predicted_probabilities, reverse=True)
    precisions, recalls = [], []
    for threshold in thresholds:
        predicted_labels = [int(p >= threshold) for p in predicted_probabilities]
        true_positives = sum([1 for (t, p) in zip(true_labels, predicted_labels) if t == 1 and p == 1])
        false_positives = sum([1 for (t, p) in zip(true_labels, predicted_labels) if t == -1 and p == 1])
        true_negatives = sum([1 for (t, p) in zip(true_labels, predicted_labels) if t == -1 and p == 0])
        false_negatives = sum([1 for (t, p) in zip(true_labels, predicted_labels) if t == 1 and p == 0])
        if true_positives + false_positives > 0:
            precisions.append(true_positives / (true_positives + false_positives))
        else:
            precisions.append(0)
        if true_positives + false_negatives > 0:
            recalls.append(true_positives / (true_positives + false_negatives))
        else:
            recalls.append(0)

    auc_score = 0
    prev_x = 0
    prev_y = 0
    for x, y in zip(recalls, precisions):
        if x != prev_x:
            auc_score += (x - prev_x) * (y + prev_y) / 2
            prev_x = x
            prev_y = y
    plt.figure()
    plt.plot(recalls, precisions, label='(AUC = {:.2f})'.format(auc_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.show()
    print("The area under the PR curve is:", auc_score)







# load true labels from file

true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter=",")
predicted_probabilities = np.genfromtxt("hw06_predicted_probabilities.csv", delimiter=",")



draw_roc_curve(true_labels, predicted_probabilities)

draw_pr_curve(true_labels, predicted_probabilities)