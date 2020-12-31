import numpy as np
import math
import operator


def calcShannonEnt(dataSet):
    dataNum=len(dataSet)
    labelDict={}
    for i in range(dataNum):
        label=dataSet[i][-1]
        labelDict[label]=labelDict.get(label,0)+1
        # print(dataSet[i][-1])
        # label=labelDict.get(dataSet[i][-1],0)
        # print(label)
        # labelDict[label]+=1
    shannonEnt=0.0
    for key in labelDict:
        prob=float(labelDict[key])/dataNum
        shannonEnt-=prob*math.log2(prob)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    # print('in split:')
    # print(dataSet)
    # # print(dataset)
    subDataSet=[]
    for data in dataSet:
        if data[axis]==value:
            # print("find me")
            reduceData=data[:axis]
            # print(reduceData)
            reduceData+=data[axis+1:] #如果axis+1刚好超了怎么办，试一下：返回空串
            # print(reduceData)
            subDataSet.append(reduceData)
    return subDataSet



def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels  # 返回数据集和分类属性

def chooseBestFeatureToSplit(dataset):
    featureNum=len(dataset[0])-1
    datasetNum=len(dataset)
    baseShannonEnt=calcShannonEnt(dataset)
    maxShannonEnt=-1
    bestFeature=-1
    for feature in range(featureNum):
        featureList=[row[feature] for row in dataset]
        featureSet=set(featureList)
        temShannonEnt=0.0
        for i in featureSet:
            subSet=splitDataSet(dataset,feature,i)
            prob=float(len(subSet))/datasetNum
            temShannonEnt+=prob*calcShannonEnt(subSet)
        infoGain=baseShannonEnt-temShannonEnt
        # print('第%d个特征的信息熵增益为%f'%(feature,infoGain))
        if infoGain > maxShannonEnt:
            maxShannonEnt=infoGain
            bestFeature=feature
    return bestFeature

def majorityCnt(labelList):
    labelDict={}
    for i in labelList:
        labelDict[i]=labelDict.get(i,0)+1
    sortedClass=sorted(labelDict.items(),operator.getitem(1),reverse=True)
    return sortedClass[0][0]

def createTree(dataSet,labels):
    # print(dataSet)
    classList=[row[-1] for row in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeature=chooseBestFeatureToSplit(dataSet)
    # print('_____________________')
    # print(labels)
    # print(bestFeature)
    featureName=labels[bestFeature]
    # print(featureName)
    del(labels[bestFeature])
    myTree={featureName:{}}
    featureSet=set([row[bestFeature] for row in dataSet])
    for val in featureSet:
        # print(val)
        myTree[featureName][val]=createTree(splitDataSet(dataSet,bestFeature,val),labels)
    return myTree

def classify(inputTree,featLabels,testVec):
    # print(inputTree.keys())
    firstStr=next(iter(inputTree))
    # print(firstStr)
    index=featLabels.index(firstStr)
    subTree=inputTree[firstStr]
    for key in subTree.keys():
        if key == testVec[index]:
            if type(subTree[key]).__name__=='dict':
                classLabel=classify(subTree[key],featLabels,testVec)
            else:
                classLabel=subTree[key]
    return classLabel

def storeTree(inputTree,fileName):
    import pickle
    with open(fileName,'wb') as fw:
        pickle.dump(inputTree,fw)

def loadTree(fileName):
    import pickle
    with open(fileName,'rb') as fr:
        tree=pickle.load(fr)
        return tree

def loadData(fileName):
    with open(fileName,'r') as fr:
        return [row.strip().split('\t') for row in fr.readlines()]

if __name__=="__main__":

    # dataset,labels=createDataSet()
    # featLabels=list(labels)
    # # myTree=createTree(dataset,featLabels)
    # # print(myTree)
    # # storeTree(myTree,'./mytree.txt')
    # myTree=loadTree('./mytree.txt')
    # print(myTree)
    # testVec = [0, 1,0,0]  # 测试数据
    # print(classify(myTree,labels,testVec))

    dataset=loadData('./lenses.txt')
    print(dataset)
