import numpy as np
import matplotlib.pyplot as plt
import random


def loadData():
    dataMat=[]
    label=[]
    with open('../data/testSet.txt','r') as fr:
        for line in fr.readlines():
            row=line.strip().split('\t')
            dataMat.append([1.0,float(row[0]),float(row[1])])
            label.append(int(row[2]))
    return dataMat,label

def showData(dataMat,label):
    fig,axs=plt.subplots(1,1)
    x0=[];y0=[]
    x1=[];y1=[]
    for i in range(len(label)):
        if label[i]==0:
            x0.append(dataMat[i][1])
            y0.append(dataMat[i][2])
        else:
            x1.append(dataMat[i][1])
            y1.append(dataMat[i][2])
    axs.scatter(x0,y0,c='red')
    axs.scatter(x1,y1,c='green')
    plt.show()

def sigmod(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataIn,label):
    dataMat=np.array(dataIn)
    labelMat=np.array(label)
    m,d=np.shape(dataMat)
    labelMat=labelMat.reshape((m,1))
    weights=np.ones((d,1))
    alpha=0.001
    maxCycle=500
    for k in range(maxCycle):
        # print(np.shape(weights))
        # print(np.shape(dataMat))
        # print(np.shape(np.dot(dataMat,weights)))
        h=sigmod(np.dot(dataMat,weights))
        # print(np.shape(h))
        # print(np.shape(labelMat))
        error=labelMat-h
        # print(np.shape(error))
        weights=weights+alpha*np.dot(dataMat.T,error)
        # print(np.shape(weights))
    return weights

def plotY(dataMat,label,weights):
    fig,axs=plt.subplots(1,1)
    x0=[];y0=[]
    x1=[];y1=[]
    for i in range(len(label)):
        if label[i]==0:
            x0.append(dataMat[i][1])
            y0.append(dataMat[i][2])
        else:
            x1.append(dataMat[i][1])
            y1.append(dataMat[i][2])
    axs.scatter(x0,y0,c='red')
    axs.scatter(x1,y1,c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    axs.plot(x,y,'blue')
    plt.show()

def gradAscent1(dataIn,label,iterNum=1500):
    dataMat=np.array(dataIn)
    # print(np.shape(dataMat))
    labelMat=np.array(label)
    m,d=np.shape(dataMat)
    labelMat=labelMat.reshape((m,1))
    weights=np.ones((d,1))
    alpha=0.005
    for k in range(iterNum):
        # alpha=alpha*0.99
        index=int(random.uniform(0,m))
        # print(np.shape(weights))
        # print(np.shape(dataMat[index]))
        h=sigmod(np.dot(dataMat[index].reshape((1,3)),weights))
        error=labelMat[index]-h
        weights=weights+alpha*np.dot(dataMat[index].reshape((3,1)),error)
    return weights

def test0():
    dataMat,label=loadData()
    print(dataMat)
    print(label)
    showData(dataMat,label)

def test1():
    dataMat, label = loadData()
    weights=gradAscent(dataMat,label)
    plotY(dataMat,label,weights=weights)

def test2():
    dataMat, label = loadData()
    weights=gradAscent1(dataMat,label)
    plotY(dataMat,label,weights=weights)

if __name__=='__main__':
    # print(1)
    test2()
