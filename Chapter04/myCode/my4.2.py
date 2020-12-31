import numpy as np
import re
import os
import random


def loadData():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for row in dataSet:
        vocabSet=vocabSet|set(row)
    retList=list(vocabSet)
    # retList.sort()
    return retList

def word2Vect(vocabList,inputData):
    outputData=np.zeros(len(vocabList))
    for word in inputData:
        if word in vocabList:
            outputData[vocabList.index(word)]=1
    return outputData

def trainNB(trainMat,label):
    dataNum=len(trainMat)
    wordNum=len(trainMat[0])
    pAbusive=sum(label)/float(dataNum)
    p0Vect=np.ones(wordNum)
    p1Vect=np.ones(wordNum)
    for i in range(dataNum):
        if label[i]==0 :
            p0Vect+=trainMat[i]
        else:
            p1Vect+=trainMat[i]
    p0Vect=np.log(p0Vect/float(sum(p0Vect)))
    p1Vect=np.log(p1Vect/float(sum(p1Vect)))
    return p0Vect,p1Vect,pAbusive

def classify( p0Vect,p1Vect,pAbusive,testMat):
    p0=sum(testMat*p0Vect)+np.log(1-pAbusive)
    p1=sum(testMat*p1Vect)+np.log(pAbusive)
    # print(p0)
    # print(p1)
    if p0>p1 :
        return 0
    else:
        return 1


def textPrase(bigString):
    retList=re.split(r'\W+',bigString)
    return [word.lower() for word in retList if len(word)>2]

# # 函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
# def createVocabList(dataSet):
#     vocabSet = set([])                      #创建一个空的不重复列表
#     for document in dataSet:
#         vocabSet = vocabSet | set(document) #取并集
#     return list(vocabSet)


def mytest0():
    docList = []
    classList = []
    hamDirs = os.listdir('../email/ham')
    print(hamDirs)
    for fname in hamDirs:
        fr = open('../email/ham/%s' % (fname), 'r')
        docList.append(textPrase(fr.read()))
        classList.append(1)
        fr.close()

    print(docList)
    print(classList)
def test0():
    docList = []; classList = []
    for i in range(1, 26):                                             #遍历25个txt文件
        wordList = textPrase(open('../email/spam/%d.txt' % i, 'r').read())#读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)                                            #标记垃圾邮件，1表示垃圾文件
        wordList = textPrase(open('../email/ham/%d.txt' % i, 'r').read()) #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)                                            #标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)                               #创建词汇表，不重复
    print(vocabList)

def test1():
    docList = []; classList = []
    testList=[];testLabel=[]
    for i in range(1, 26):                                             #遍历25个txt文件
        wordList = textPrase(open('../email/spam/%d.txt' % i, 'r').read())#读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)                                            #标记垃圾邮件，1表示垃圾文件
        wordList = textPrase(open('../email/ham/%d.txt' % i, 'r').read()) #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)                                            #标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)                               #创建词汇表，不重复
    trainData=[]
    for row in docList:
        trainData.append(word2Vect(vocabList,row))
    for i in range(10):
        randomIndex=int(random.uniform(0,len(trainData)))
        testList.append(trainData[randomIndex])
        testLabel.append(classList[randomIndex])
        del(trainData[randomIndex])
        del(classList[randomIndex])
    p0Vect, p1Vect, pAbusive=trainNB(trainData,classList)
    i=0
    j=0
    for row in testList:
        if classify(p0Vect,p1Vect,pAbusive,row)!=testLabel[i]:
            print('第%d错了'%(i))
            j+=1
        i+=1
    print(j/10.0)


if __name__=='__main__':
    test1()




