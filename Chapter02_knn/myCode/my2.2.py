import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import operator


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

def file2matrix(filename):
    with open(filename, 'r') as fr:
        # print(fr.readline())
        datingArray = fr.readlines()
        numberOfLines= len(datingArray)
        returnMat=np.zeros((numberOfLines,3))
        # print(returnMat.shape)
        label=[]
        index=0
        labelDict={'didntLike':1,'smallDoses':2,'largeDoses':3}
        # print('this is line')
        # print('this is line')
        # print(datingArray[0])
        # print('this is line')
        # print('this is new line')
        # print('this is new line')
        # print(datingArray[0].strip())
        # print('this is new line') 果然把换行符去掉了
        # print(datingArray[0].strip().split('\t')) 数据确实是用tab做分割的，只不过数字太长了看不出来而已
        for line in datingArray:
            lineList = line.strip().split('\t')
            returnMat[index,:]=lineList[0:3]
            label.append(labelDict[lineList[-1]])
            index+=1
        return returnMat,label

def showdData(datingDataMat,datingLabels):
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    labelColour=[]
    for i in datingLabels:
        if i==1:
            labelColour.append('black')
        elif i== 2:
            labelColour.append('orange')
        elif i==3:
            labelColour.append('red')
    axs[0][0].scatter(datingDataMat[:,0],datingDataMat[:,1],color=labelColour)
    axs[0][0].set_title('fly and game')
    axs[0][0].set_xlabel('fly')
    axs[0][0].set_ylabel('game')

    axs[0][1].scatter(datingDataMat[:, 0], datingDataMat[:, 2], color=labelColour)
    axs[0][1].set_title('fly and eat')
    axs[0][1].set_xlabel('fly')
    axs[0][1].set_ylabel('eat')

    axs[1][0].scatter(datingDataMat[:, 1], datingDataMat[:, 2], color=labelColour)
    axs[1][0].set_title('game and eat')
    axs[1][0].set_xlabel('game')
    axs[1][0].set_ylabel('eat')

    # 测试散点图
    # axs[1][1].scatter(range(len(labelColour)),datingDataMat[:, 1], color='green')

    axs[1][1].plot(datingDataMat[:50, 1], color='green')
    # 添加图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])

    plt.show()

def autoNormal(dataingMat):
    maxVal=dataingMat.max(0)
    minVal=dataingMat.min(0)
    nrow=dataingMat.shape[0]
    ranges=maxVal-minVal

    normalData=(dataingMat-minVal)/np.tile(ranges,(nrow,1))
    return normalData,ranges,minVal

def datingClassTest():
    datingMat, datingLab = file2matrix("../Ch02/datingTestSet.txt")
    normalData,range,minval=autoNormal(dataingMat=datingMat)
    ratio=0.9
    train_num=int(normalData.shape[0]*ratio)
    test_num=normalData.shape[0]-train_num
    errorCount=0
    for i in range(test_num):
        ilabel=classify0(normalData[i,:],normalData[test_num:,:],datingLab[test_num:],4)
        # print('第'+i+'个数据：分类结果：'+ilabel+'\t实际标签：'+datingLab[i])

        if ilabel!=datingLab[i]:
            errorCount+=1
            print('第%d个数据：分类结果为%d\t实际结果为%d' % (i, ilabel, datingLab[i]))
    print('错误率：%f%%'%(100*errorCount/test_num))

def classifyPerson():
    mails=float(input('飞行公里数:'))
    precent=float(input('百分比:'))
    iceCream=float(input('冰激凌:'))
    person=np.array([mails,precent,iceCream])
    fDict={1:'讨厌',2:'有点喜欢',3:'很喜欢'}

    datingMat, datingLab = file2matrix("../Ch02/datingTestSet.txt")
    normalData, range, minval = autoNormal(dataingMat=datingMat)
    person=(person-minval)/range
    label=classify0(person,normalData,datingLab,3)

    print('你很可能%s这个人'%(fDict[label]))



if __name__=='__main__':
    # datingClassTest()
    # print(datingMat)
    # print(datingLab)
    # showdData(datingMat,datingLab)
    classifyPerson()
