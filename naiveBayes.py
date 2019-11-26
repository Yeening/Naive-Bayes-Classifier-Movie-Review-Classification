import sys
import os
import numpy as np
import nltk
import math
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize

nltk.download('stopwords')
###############################################################################


def buildVocabulary(Path):
    tokenizer = RegexpTokenizer(r'\w+') #tokenizer removing puncturations
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english')) 
    tokens = []
    for folder in os.listdir(Path):
        for filename in os.listdir(Path+folder):
            data = open(Path+folder+'/'+filename, 'r')
            raw = data.read()
            token = tokenizer.tokenize(raw)
            # stop words removal
            filtered_token = [w for w in token if not w in stop_words]
            # stemming
            for i in range(len(filtered_token)):
                filtered_token[i] = ps.stem(filtered_token[i])
            tokens.extend(filtered_token)
    word_counts = Counter(tokens).most_common(3999)
    vocab = {}
    i = 0
    for word_count in word_counts:
        vocab[word_count[0]] = i
        i += 1
    vocab['UNK'] = i
#     output_f = open('m_dic.txt', 'w')
#     words = [row[0]+'\n' for row in word_counts]
#     words.append('UNK')
#     output_f.writelines(words)
    return vocab


def loadData(Path, vocabulary):
    Xtrain, Xtest = [], []
    ytrain, ytest = [], []
    for filename in os.listdir(Path+'training_set/pos/'):
        Xtrain.append(transfer(Path+'training_set/pos/'+filename, vocabulary))
        ytrain.append(1)
#         data = open(Path+'training_set/pos/'+filename,'r')
#         for line in data:
#             Xtrain.append(line[:-2])
#             ytrain.append(1)
    for filename in os.listdir(Path+'training_set/neg/'):
        Xtrain.append(transfer(Path+'training_set/neg/'+filename, vocabulary))
        ytrain.append(0)
#         data = open(Path+'training_set/neg/'+filename,'r')
#         for line in data:
#             Xtrain.append(line[:-2])
#             ytrain.append(0)
    for filename in os.listdir(Path+'test_set/pos/'):
        Xtest.append(transfer(Path+'test_set/pos/'+filename, vocabulary))
        ytest.append(1)
#         data = open(Path+'test_set/pos/'+filename,'r')
#         for line in data:
#             Xtest.append(line[:-2])
#             ytest.append(1)
    for filename in os.listdir(Path+'test_set/neg/'):
        Xtest.append(transfer(Path+'test_set/neg/'+filename, vocabulary))
        ytest.append(0)
#         data = open(Path+'test_set/neg/'+filename,'r')
#         for line in data:
#             Xtest.append(line[:-2])
#             ytest.append(0)    
    Xtrain, Xtest = np.array(Xtrain), np.array(Xtest)
    ytrain, ytest = np.array(ytrain), np.array(ytest)
    return Xtrain, Xtest, ytrain, ytest


def transfer(fileDj, vocabulary):
    data = open(fileDj,'r')
    BOWDj = np.zeros((len(vocabulary.keys()),), dtype = int)
    
    tokenizer = RegexpTokenizer(r'\w+') #tokenizer removing puncturations
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english')) 
    
    raw = data.read()
    tokens = tokenizer.tokenize(raw)
    # stop words removal
    filtered_token = [w for w in tokens if not w in stop_words]
    # stemming
    for i in range(len(filtered_token)):
        filtered_token[i] = ps.stem(filtered_token[i])
    # transfer to BOW
    for token in filtered_token:
        if token in vocabulary:
            BOWDj[vocabulary[token]] += 1
        else:
            BOWDj[vocabulary['UNK']] += 1
    
    return BOWDj


def naiveBayesMulFeature_train(Xtrain, ytrain):
    thetaPos = np.zeros((Xtrain.shape[1],), dtype = float)
    thetaNeg = np.zeros((Xtrain.shape[1],), dtype = float)
    for i in range(len(Xtrain)):
#         Binary = np.where(Xtrain[i] > 0, 1, 0)
        if ytrain[i] == 1:
            thetaPos += Xtrain[i]
        elif ytrain[i] == 0:
            thetaNeg += Xtrain[i]
    thetaPos = (thetaPos+1)/(np.sum(thetaPos)+Xtrain.shape[1])
    thetaNeg = (thetaNeg+1)/(np.sum(thetaNeg)+Xtrain.shape[1])
#     thetaPos += 1
#     thetaNeg += 1
#     thetaPos /= (Xtrain.shape[0]/2 + 2)
#     thetaNeg /= (Xtrain.shape[0]/2 + 2)
    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    yPredict = []
    for X in Xtest:
        N = np.sum(X) #total words of X
        thetaLogPos = 0.0
        thetaLogNeg = 0.0
        for i in range(len(X)):
            if X[i] > 0:
                if thetaPos[i] > 0:
                    thetaLogPos += X[i] * math.log2(thetaPos[i])
                if thetaNeg[i] > 0:
                    thetaLogNeg += X[i] * math.log2(thetaNeg[i])
        if thetaLogNeg > thetaLogPos:
            yPredict.append(0)
        else:
            yPredict.append(1)
    yPredict = np.array(yPredict)
#     print(yPredict, ytest)
    errors = np.sum(np.absolute(yPredict - ytest))
    Accuracy = (ytest.shape[0] - errors)/ytest.shape[0]
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    yPredict = clf.predict(Xtest)
    errors = np.sum(np.absolute(yPredict - ytest))
    Accuracy = (ytest.shape[0] - errors)/ytest.shape[0]
    return Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
    thetaPosTrue = np.zeros((Xtrain.shape[1],), dtype = float)
    thetaNegTrue = np.zeros((Xtrain.shape[1],), dtype = float)
    for i in range(len(Xtrain)):
        Binary = np.where(Xtrain[i] > 0, 1, 0)
        if ytrain[i] == 1:
            thetaPosTrue += Binary
        elif ytrain[i] == 0:
            thetaNegTrue += Binary
    thetaPosTrue += 1
    thetaNegTrue += 1
    thetaPosTrue /= (Xtrain.shape[0]/2 + 2)
    thetaNegTrue /= (Xtrain.shape[0]/2 + 2)
    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    for X in Xtest:
        N = np.sum(X) #total words of X
        thetaLogPos = 0.0
        thetaLogNeg = 0.0
        for i in range(len(X)):
            if X[i] > 0:
                if thetaPos[i] > 0:
                    thetaLogPos += math.log2(thetaPosTrue[i])
                if thetaNeg[i] > 0:
                    thetaLogNeg += math.log2(thetaNegTrue[i])
        if thetaLogNeg > thetaLogPos:
            yPredict.append(0)
        else:
            yPredict.append(1)
    yPredict = np.array(yPredict)
#     print(yPredict, ytest)
    errors = np.sum(np.absolute(yPredict - ytest))
    Accuracy = (ytest.shape[0] - errors)/ytest.shape[0]
    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    
    dic = buildVocabulary(textDataSetsDirectoryFullPath+'/training_set/')

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath, dic)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

 
    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")





