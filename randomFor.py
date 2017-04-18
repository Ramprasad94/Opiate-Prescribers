#!usr/bin/python

#program to implement bagging and boosting
import random
import os, sys
import math
import copy

from pandas import DataFrame
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import time
import numpy as np
import matplotlib.pyplot as plt
#START OF CLASS DECISION NODE
class Decision_node:								# class to represent each node in the tree
    def __init__(self, results=None,depthLevel = 1,col=None,values=None,children=[],hasChildren = False):  #initialize each node in the decision tree
        self.results = results          # a list of lists to store the resulting rows
        self.col = col                  # a variable to store the column value of the attribute to be split on
        self.children = children        # a list containing the children to each node
        self.depthLevel = depthLevel    # height till which the tree has to be constructed
        self.isLeaf = False             # a variable to test if the node is leaf
        self.parent = None              # a variable to keep track of the parent of the current node
        self.classDist = None           # a variable to give out the class distribution of a particular node
        self.colValues = None

    #This method splits given results set based on a feature
    def splitData(self):
        resultSet = {} # a dictionary to store the rows associated to each attribute value

        for result in self.results:
            if result[self.col] not in resultSet:
                resultSet[result[self.col]] = [result]
            else:
                resultSet[result[self.col]].append(result)

        return resultSet

    #This method sets the leaf nodes of decission tree to some class : (0,1)
    def setClassDist(self): # a method to store the class distribution for a node based on majority values
        results = self.results

        count0 = 0
        count1 = 0
        for result in results:
            if (result[-1] == 0):  # @change: index is the class label index
                count0 += 1
            elif (result[-1] == 1): # @change: index is the class label index
                count1 += 1

        if (count0 > count1):
            self.classDist = 0
        else:
            self.classDist = 1

    #This method classifies a given test record to either : (0/1)
    def classify(self,testRecord):
        if(self.isLeaf):
            return self.classDist
        else:
            col = self.col
            for child in self.children:
                if(child.results[0][col] == testRecord[col]):
                    return child.classify(testRecord)

    #This method organizes the decision tree
    def deleteExtraChildren(self):
        result = []
        for i in range(len(self.children)):
            if(self.children[i].parent == self):
                result.append(self.children[i])
        self.children = result

#END OF CLASS DECISION NODE

def class_attrib_value_count(results):  # a function to give out the existing class distributions of a given dataset.
    count_dict = {}   # a dictionary to maintain count of each attribute value
    for row in results:
        value = row[-1] #@change : index is class label
        if value in count_dict:    # if value is already in dict, increment it
            count_dict[value] += 1
        else:
            count_dict[value] = 1   # else assign its count as zero
    return count_dict

def entropy(results):       #a function to calculate the entropy of a particular dataset
    entropy_value = 0.0
    rows_length = len(results)
    counted_dict = class_attrib_value_count(results)
    for value in counted_dict.keys():
        p = float(counted_dict[value])/rows_length
        if p<=0:
            p=1
        else:
            entropy_value -= (p * math.log(p,2))
    return entropy_value

#This method finds if a given dataset is pure or not i.e., is it all from same class - (0/1)
def isImPure(results):
    count0=0
    count1=0
    for result in results:
        if(result[-1]==0): # @change: index is the class label index
            count0 +=1
        elif(result[-1]==1): # @change: index is the class label index
            count1+=1
        if(count0>0 and count1>0):
            return True
    return False


#START OF TREE BUILDING RECURSIVE FUNCTION : This method recursively builds a decision tree for a given dataset , feature list and a  depth
def buildTree(results,totalDepth,featureList,initialDepth,parent = None):
    newNode = Decision_node(results, initialDepth)
    newNode.parent = parent
    best_gain = 0
    best_attrib = None
    best_partition= None
    current_entropy = entropy(results) # find out the entropy of the new node containing the subset
    for column in featureList:
        newNode.col = column
        partitions = newNode.splitData()  # split up the node into their resulting children along with their own subsets

        new_entropy = 0.0 # set the intermediate entropy computation to zero
        for val in partitions: # loop through all the possible column values
            new_entropy = new_entropy + (entropy(partitions[val]) * (float(len(partitions[val]))/len(results)) ) # calculate the weighted entropy for that column
        information_gain = current_entropy - new_entropy
        if (information_gain > best_gain):
            best_gain = information_gain
            best_attrib = column
            best_partition = partitions

    newNode.col = best_attrib # set the column with highest information gain(best attribute) to be the splitting column
    if(newNode.depthLevel<=totalDepth and len(results)>1 and isImPure(results) and best_attrib!=None) :
        resultSet = best_partition
        newNode.colValues=resultSet.keys()
        for i in resultSet:
            x = buildTree(resultSet[i],totalDepth,featureList,initialDepth+1,newNode)
            if x.depthLevel == newNode.depthLevel+1:
                newNode.children.append(x)
    else:
        newNode.isLeaf = True
        newNode.children = []
        newNode.setClassDist()

    newNode.deleteExtraChildren()
    return newNode

#END OF TREE BUILDING RECURSIVE FUNCTION


def load_data(filepath):
    data = []
    file = open(filepath, "r")
    lines = file.readlines()
    for line in lines[1:]:
        newline = line.strip().split(",")
        for i in range(len(newline)):
            if(newline[i].isdigit()):
                newline[i] = int(newline[i])
        #newline = map(int, newline)
        data.append(newline)
    file.close()
    print "loaded the file"
    print len(data),len(data[0])
    return data

def doFeatureEngineering(train_data):
    tv = []
    x = []
    for row in train_data:
        tv.append(row[-1])
        x.append(row[5:-1])
    nptD = x
    sel = VarianceThreshold(threshold=(0.1 * (1 - 0.1)))
    sel.fit_transform(nptD)

    #### Below is chi sqaured test:
    tvnp = np.array(tv)
    nptDNew = SelectKBest(chi2, int(len(nptD[0]) * 0.95) ).fit_transform(nptD, tvnp)

    tData = []

    for i in range(len(nptDNew)):
        row = [x for x in nptDNew[i]]
        tData.append(row+[tv[i]])

    return tData

def learn_bagged(tdepth, nummodels, filepath):
    train_data = [] #declare empty lists to store all the training records and test records
    test_data = []

    print "hi.. This program takes about 2 to 3 minutes to display the output"
    train_data = load_data(filepath)
    tData = []

    for row in train_data:
        if len(row) == 256:
            tData.append(row)

    train_data = tData

    train_data = doFeatureEngineering(train_data)
    l = int(len(train_data) * 0.9)
    test_data = train_data[l:]
    train_data = train_data[:l]
    print "Number of records in train_data = "+str(len(train_data))
    print "Number of records in test = " + str(len(test_data))

    # @change: compute the feature indices
    featureList = [i for i in range(len(train_data[0]))]
    featureList = featureList[:-1]
    print "len of feature List=",len(featureList)

    totalDepth = tdepth
    #loop to create  bootstrap samples according to value given in nummodels
    samples = []
    for i in range(1,nummodels+1):
        temp_list = []
        for k in range(int(len(train_data) * 0.8)):
            number = random.randrange(0,len(train_data))
            temp_list.append(train_data[number])
        samples.append(temp_list)

    print "Number of bags = "+str(len(samples))
    head = []
    for bootstrap in samples:
        head.append(buildTree(bootstrap,totalDepth, featureList, 1)) # create required number of decision trees and append it to head list.
    incorrectly_classified = 0
    correctly_classified = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for t in test_data:
        predictedList = []
        tcopy = copy.deepcopy(t)
        tcopy[-1] = None
        for i in range(len(head)):
            predictedList.append(head[i].classify(tcopy))

        predicted = max(set(predictedList),key = predictedList.count)
        if(predicted != t[-1]):    # @change: index is the class label index
            incorrectly_classified += 1
        else:
            correctly_classified +=1

        if (predicted == 1 and t[-1] == 1): # @change: index is the class label index
            tp += 1
        elif (predicted == 0 and t[-1] == 1):# @change: index is the class label index
            fn += 1
        elif (predicted == 1 and t[-1] == 0): # @change: index is the class label index
            fp += 1
        elif (predicted == 0 and t[-1] == 0): # @change: index is the class label index
            tn += 1
    acc = calculate_accuracy(incorrectly_classified, correctly_classified)
    create_confusion_matrix(tp, fn, fp, tn)

#Finds the accuracy
def calculate_accuracy(incorrectly_classified,correctly_classified):

    print("\n\n\nIncorrectly classified= " + str(incorrectly_classified) + "\t\t Correctly classified= " + str(correctly_classified)+"\n")
    accuracy = float(correctly_classified) / (correctly_classified + incorrectly_classified)
    print("\nAccuracy for a depth of " + str(tdepth) + " is " + str(accuracy*100)+" %"+"\n")
    return accuracy

#This method prints the confusion matrix
def create_confusion_matrix(tp,fn,fp,tn):
    print "\nThe confusion matrix is as follows:\n"
    print "                                 Predicted"
    print "----"*20
    print "Actual"
    print "                 True Negative: "+str(tn),
    print "                 False Positive: "+str(fp)
    print "\n"
    print "                 False Negative: "+str(fn),
    print "                 True Positive: "+str(tp)
    print "----"*20


if __name__ == "__main__":
    time1 = time.time()
    # Get the ensemble type
    entype = "bag"
    # Get the depth of the trees
    tdepth = 5
    # Get the number of bags or trees
    nummodels = 5
    # Get the location of the data set
    #datapath = sys.argv[1]
    datapath = "/Users/hannavaj/Desktop/FP"
    datapath += "/prescriber-info.csv"

    learn_bagged(tdepth, nummodels, datapath);

