import numpy as np

from numpy import *
import csv
import matplotlib.pyplot as plt
import random

def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fileObject:
        reader = csv.DictReader(fileObject)
        listClass = []
        labelName = list(reader.fieldnames) #存入文件头
        num = len(labelName) - 1

        for line in reader.reader:
            dataMat.append(line[:num])
            if line[-1] == 'male': 
                gender = 1
            else:
                gender = 0
            listClass.append(gender)

        dataMat = np.array(dataMat).astype(float) 
        countVector = np.count_nonzero(dataMat, axis = 0)
        sumVector = np.sum(dataMat, axis=0) 
        meanVector = sumVector/countVector
        #求得每个特征的平均值

        for row in range(len(dataMat)):
            for col in range(num):
                if dataMat[row][col] == 0.0:
                    dataMat[row][col] = meanVector[col]
        #把缺失项用平均值填入
        
        minVector = dataMat.min(axis = 0)
        maxVector = dataMat.max(axis = 0)
        diff = maxVector - minVector
        diff = diff/9

        dataSet = []
        for i in range(len(dataMat)):
            line = np.array((dataMat[i] - minVector)/diff).astype(int)
            dataSet.append(line)

        #构造训练集和测试集



        trainSet = random.sample(range(0, 3167), 2218)  # 设置随机数生成从0-3167中随机挑选2218个随机数

        testSet = list(set(range(0, 3167)).difference(set(trainSet)))

        #将数据集以7：3比例划分

    #训练集
        trainMat = []
        trainClasses = []
        for i in trainSet:
            trainMat.append(dataSet[i])
            trainClasses.append(listClass[i])

        #测试数据集

        testMat = []
        testClasses = []
        for i in testSet:
            testMat.append(dataSet[i])
            testClasses.append(listClass[i])

    return trainMat, trainClasses, testMat, testClasses, labelName


def bayes(trainMatrix, listClasses):
    numTrainData = len(trainMatrix)
    numFeature = len(trainMatrix[0])

    p1class = sum(listClasses) / float(numTrainData)

    n = 10
    listClasses_1 = []
    trainData_1 = []

    for i in list(range(numTrainData)):
        if listClasses[i] == 1:
            listClasses_1.append(i)
            trainData_1.append(trainMatrix[i])

    #分类为1情况下各特征概率
    trainData_1 = np.matrix(trainData_1)
    p1feature = {}
    for i in list(range(numFeature)):
        featureValues = np.array(trainData_1[:, i ]).flatten()
        featureValues = featureValues.tolist() + list(range(n))
        p = {}
        count = len(featureValues)
        for value in set(featureValues):
            p[value] = np.log(featureValues.count(value)/float(count)) #
        p1feature[i] = p

    #所有分类下各特征概率
    pfeature = {}
    trainMatrix = np.matrix(trainMatrix)
    for i in list(range(numFeature)):
        featureValues = np.array(trainMatrix[:,i]).flatten()
        featureValues = featureValues.tolist() + list(range(n))
        p = {}
        count = len(featureValues)
        for value in set(featureValues):
            p[value] = np.log(featureValues.count(value)/float(count))
        pfeature[i] = p
    
    return pfeature, p1feature, p1class

def prediction(testVector, pfeature, p1feature, p1class):
    sum = 0.0
    for i in list(range(len(testVector))):
        sum = sum + p1feature[i][testVector[i]]
        sum = sum - pfeature[i][testVector[i]]
    p1 = sum + np.log(p1class)
    p0 = 1 - p1
    if p1 > p0:
        return 1
    else: 
        return 0

def test():
    filename = 'D:/voice.csv'
    trainMat, trainClasses, testMat, testClasses, labelName = loadDataSet(fileName=filename)

    pfeature, p1feature, p1class = bayes(trainMat, trainClasses)

    count_1 = 0
    correctCount_1 = 0
    count_0 = 0
    correctCount_0 = 0

    for i in list(range(len(testMat))):
        testVector = testMat[i]
        result = prediction(testVector, pfeature, p1feature, p1class)
        if testClasses[i] == 1:
            count_1 = count_1 + 1
            if(result == 1):
                correctCount_1 = correctCount_1 + 1
        
        if testClasses[i] == 0:
            count_0 = count_0 + 1
            if(result == 0):
                correctCount_0 = correctCount_0 + 1

    return count_0, correctCount_0, count_1, correctCount_1

def picture(rate, number, title, Yname):
    x = range(1, number + 1)
    plt.plot(x, rate)
    plt.xlabel("Test times")
    plt.ylabel(Yname)
    plt.title(title)
    plt.show()

listNumber = 20;
rate0 = []
rate1 = []
for i in range(listNumber):
    tempc0, tempcc0, tempc1, tempcc1 = test()
    rate0.append(float(tempcc0/tempc0))
    rate1.append(float(tempcc1/tempc1))

print("Male voice correct rate:     ", mean(rate1))
print("Female voice correct rate:   ", mean(rate0))

picture(rate0, listNumber, "Female voice correct rate", "correct rate")
picture(rate1, listNumber, "Male voice correct rate", "correct rate")
      
