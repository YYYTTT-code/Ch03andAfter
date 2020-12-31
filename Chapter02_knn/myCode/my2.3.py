import numpy as np
import os
import operator


def img2vector(filename):
    returnVec = np.zeros((1,1024))
    with open(filename) as file:
        for i in range(32):
            # returnVec[1,i*32:(i+1)*32]=file.readline()
            str=file.readline()
            # print(str)
            # print(len(str)) 33 因为带了结尾的换行符
            # for j in len(str):
            for j in range(32):
                returnVec[0,i*32+j]=int(str[j])
        return returnVec

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape
    subMat=np.tile(inX,(dataSetSize[0],1))-dataSet
    sqrMat=subMat**2
    sumMat=sqrMat.sum(axis=1)
    disMat=sumMat**0.5

    sortedDis=disMat.argsort()
    classCount={}
    for i in range(k):
        temlabel=labels[sortedDis[i]]
        classCount[temlabel]=classCount.get(temlabel,0)+1

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def handwritingClassTest():
    hwLabels=[]
    trainDirs=listdir('../Ch02/digits/trainingDigits')
    testDirs=listdir('../Ch02/digits/testDigits')
    m=len(trainDirs)
    trainMat=np.zeros((m,1024))
    for i in range(m):
        hwLabels.append(int(trainDirs[i].split('_')[0]))
        trainMat[i,:]=img2vector('../Ch02/digits/trainingDigits/{}'.format(trainDirs[i]))

    n=len(testDirs)
    errorNum=0
    for i in range(n):
        realLabel=int(testDirs[i].split('_')[0])
        inX=img2vector('../Ch02/digits/testDigits/{}'.format(testDirs[i]))
        testLabel=classify0(inX,trainMat,hwLabels,3)
        if testLabel!=realLabel:
            errorNum+=1
            print('真正的类别是：%s\t训练的类别是：%s'%(realLabel,testLabel))
    print('共错误%d次，错误率为%f'%(errorNum,errorNum/n))




if __name__=='__main__':
    # vector=img2vector('../Ch02/digits/trainingDigits/0_0.txt')
    # print(vector.shape)
    # print(vector[0,:64])
    # handwritingClassTest()
    dir=os.listdir("../Ch02/digits/")
    print(dir)
    print(os.path.isdir("../Ch02/digits/"))
    print(os.path.isfile("../Ch02/digits/"))