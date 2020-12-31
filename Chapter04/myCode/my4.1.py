import numpy as np


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



def test0():
    postingList, classVec = loadData()
    print('postingList:\n', postingList)
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(word2Vect(myVocabList, postinDoc))
    print('trainMat:\n', trainMat)
    # 每次输出的向量矩阵不太一样，是set转换成list时不能保证顺序导致，可以加个sort，不过没必要
    # print(type(trainMat))
    # print(trainMat[1][3])

def test1():
    postingList, classVec = loadData()
    print('postingList:\n', postingList)
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(word2Vect(myVocabList, postinDoc))
    print('trainMat:\n', trainMat)
    # 每次输出的向量矩阵不太一样，是set转换成list时不能保证顺序导致，可以加个sort，不过没必要
    # print(type(trainMat))
    # print(trainMat[1][3])
    # a=np.array([1,0,2])
    # b=a+a
    # print(b)
    p0V, p1V, pAb = trainNB(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)

def test2():
    postingList, classVec = loadData()
    print('postingList:\n', postingList)
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(word2Vect(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)
    testEntry = ['stupid', 'my', 'worthless']
    thisDoc = np.array(word2Vect(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classify(p0V, p1V, pAb,thisDoc))

if __name__=='__main__':
    test2()