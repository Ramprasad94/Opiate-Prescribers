'''
Spam classification
'''
import math
import os
import time
import pickle
import copy
import csv
import pandas
import sys

def create_confusion_matrix(tp,fn,fp,tn):
    print "\nThe confusion matrix for Naive Bayes is as follows:\n"
    print "                                 Predicted"
    print "----"*20
    print "Actual"
    print "                 True Negative: "+str(tn),
    print "                 False Positive: "+str(fp)
    print "\n"
    print "                 False Negative: "+str(fn),
    print "                 True Positive: "+str(tp)
    print "----"*20

#reads data from files in a folder and appends them to a list of strings
def read_data_NVB(datapath):
    postProb = {0:0.0,1:0.0}
    lines = []

    df = pandas.read_csv(datapath+ "/prescriber-info.csv")
    lines = df.values

    l = []
    for row in lines:
        l.append([x for x in row])
    lines = l

    featureNames = lines[0]

    lines = lines[1:]
    count0 = 0
    count1 = 0
    for j in range(len(lines)):
        if lines[j][-1] == 0:
            count0 += 1
        elif lines[j][-1] == 1:
            count1 += 1


    postProb[0] = float(count0) / (count0 + count1)
    postProb[1] = float(count1) / (count0 + count1)

    return lines,postProb,featureNames

# totalWordsDict = {} # {"w1":0,"w2":1,.....}
def computeLikelihoodDicts(trainData,featureNames):
    print "entered computing likehiloods "
    lk = { 0:{}, 1: {} }
    total =  {0:0, 1:0}

    for row in trainData:
        key = row[-1]
        for j in range(5,len(row[:-1])):
            word = featureNames[j]
            if(word in lk[key]):
                lk[key][word] += row[j]
            else:
                lk[key][word] = row[j]
            total[key] += row[j]

    for type in lk:
        for word in lk[type] :
            if lk[type][word] == 0:
                lk[type][word] = 1
            lk[type][word] = float (lk[type][word]) / total[type]
            lk[type][word] = math.log(lk[type][word])

    return lk

### MAIN SECTION #####
time1 = time.time()
print "Welcome to Naive Bayes Classifier"
print "This program takes about 10 seconds to show the output"
datapath = sys.argv[1]
trainData,postProb,featureNames = read_data_NVB(datapath)
tData = []
l = 0.9
l = int(len(trainData) * 0.9)
testData = trainData[l:]
trainData = trainData[:l]



lk = computeLikelihoodDicts(trainData,featureNames)
corr = 0
wrong = 0
count0 = 0
count1 = 0

testDataCopy = copy.deepcopy(testData)
for row in testData:
    row[-1] = None

tp,fn,fp,tn = 0,0,0,0

for i in range(len(testData)):
    prob0 = 0
    prob1 = 0
    row = testData[i]
    for j in range(5,len(row)-1):
        if (row[j] != 0):
            word = featureNames[j]
            if(word in lk[0] and word in lk[1]):
                prob0 += lk[0][word]
                prob1 += lk[1][word]

    prob0 += math.log(postProb[0])
    prob1 += math.log(postProb[1])

    if(prob1 > prob0):
        cls = 1
        count1+=1
    else:
        cls = 0
        count0 += 1

    if(cls == testDataCopy[i][-1]):
        corr += 1
    else:
        wrong += 1

    if (cls == 1 and testDataCopy[i][-1] == 1):  # @change: index is the class label index
        tp += 1
    elif (cls == 0 and testDataCopy[i][-1] == 1):  # @change: index is the class label index
        fn += 1
    elif (cls == 1 and testDataCopy[i][-1] == 0):  # @change: index is the class label index
        fp += 1
    elif (cls == 0 and testDataCopy[i][-1] == 0):  # @change: index is the class label index
        tn += 1
print "corr = ",corr , "wrong =",wrong

create_confusion_matrix(tp,fn,fp,tn)
print "accuracy=",float(corr) * 100 / (corr + wrong) , "%"

print "time taken=",time.time()-time1
