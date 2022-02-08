import xml.dom.minidom as xdm
from xml.dom.minidom import parse
import re
import time
#from textblob import TextBlob
import pandas as pd
import numpy as np
import sys

sys.__stdout__=sys.stdout 

start_time = time.time()

def ExtractTopicNames(fileName):
    f = open(fileName)                      
    LIST = []
    for line in f:
        row = line[:-1]
        LIST.append(row)           
    f.close()
    return LIST

'''def SpellCheck(DICT):
    for WORD in DICT:
        print(1)
        WORD = TextBlob(WORD)
        WORD = str(WORD.correct())
    return DICT'''

def MakeStopWordDoc(stopwordsfile):
    f = open(stopwordsfile)
    ListOfStopWords = []
    ListOfStopWords = ExtractTopicNames(stopwordsfile)
    f.close()
    return ListOfStopWords

def CheckStopWord(word):
    if word in stop_word_list:
        a = 'YES'
        return a
    else:
        a = 'NO'
        return a
          
def UpdateDataframe(df,max_rows,dictionary,topics_xml):
    index_count = 0
    for i in range(len(topics_xml)):
        #print(topics_xml[i])
        DOMTree = xdm.parse(topics_xml[i])
        examples = DOMTree.getElementsByTagName("row")
        c = 0
        label = labels[i]
        for row in examples:
            if c == (max_rows):
                break
            body = row.getAttribute("Body")
            body = body.lower()
            doc = re.findall(r'\w+',body)
            df.loc[index_count,'labels'] = label
            if len(doc)<5:
                continue
            for word in doc:
                if word in dictionary:
                    col = dictionary[word]
                    df.loc[index_count,col]+=1
            index_count = index_count+1
            c = c+1
    return df

def MakeDataframe(max_rows,dictionary):
    no_of_index = max_rows*len(labels)
    indices = list(range(0,no_of_index))
    column = list(range(0,len(dictionary)))###used for column names in dataframe, helps identify the position of a certain word in a dictionary
    label = ['labels']
    column = column + label   
    df = pd.DataFrame(0,index = indices,columns = column)
    return df    

def LoopForK(dist_df_list):
    ACC = []
    for val in range(len(k_values)):
            for i in range(len(_test_vectors)):
                dist_df1 = dist_df_list[i]
                lb=dist_df1[:k_values[val]]
                lb=lb[1]
                lb=lb.mode()
                lb=lb.values[0]
                test_df.loc[i, 'result'] = lb    
                #test_df.loc[i, 'distance'] = DIST
            accuracy = FindAccuracy(test_df)
            ACC.append(accuracy)
    return ACC

def MeasureHammingDistance(test_vect,train_vect):
    test_vect=np.array(test_vect>0,dtype=np.int8)
    train_vect=np.array(train_vect>0,dtype=np.int8)
    dist=train_vect-test_vect
    dist=np.abs(dist)
    distance=np.sum(dist)
    return distance

def MeasureEucledianDistance(test_vect,train_vect):
    dist=np.abs(test_vect-train_vect)
    dist=np.square(dist)
    dist=np.sum(dist)
    dist=np.sqrt(dist)
    return dist

def VectorWithTFIDF(test_vect,_vectors):
    no_of_total_w = np.sum(test_vect)
    vect_for_cosine = []
    for i in range(len(test_vect)):
        if (test_vect[i] == 0.0):
            TFIDF = 0.0
            vect_for_cosine.append(TFIDF)
        else:
            TF = (test_vect[i]/no_of_total_w)
            Cw = weight_vect[i]
            s = D/Cw
            IDF = np.log(s)
            TFIDF = TF*IDF
            vect_for_cosine.append(TFIDF)
    vect_for_cosine = np.array(vect_for_cosine, dtype = np.float64)
    return vect_for_cosine
        
def MeasureCosineSimilarity(test_vect,train_vect):
    temp=np.square(test_vect)
    temp=np.sum(temp)
    norm_test=np.sqrt(temp)
        
    temp=np.square(train_vect)
    temp=np.sum(temp)
    norm_train = np.sqrt(temp)
        
    numerator=np.sum(test_vect*train_vect)
    denominator=norm_test*norm_train
        
    similarity = numerator/denominator

    return similarity
def FindAccuracy(test_df):
    accuracy = 0
    for i in range(len_test_df):
       if test_df.loc[i,'labels']==test_df.loc[i,'result']:
            accuracy=accuracy+1
    accuracy=accuracy/(len_test_df)
    #accuracy = accuracy*100
    return accuracy
def ApplyNaiveBayes(train_dataframe,test_dataframe,alpha):
    grouped = df.groupby('labels')
    VECTOR_PROBWCm = []
    PCm = (1/no_of_topics)
    for name in labels:
        current = grouped.get_group(name)
        tempo_vectors=np.array(current.loc[:,col_names],dtype=np.float64)
        NwCm = np.sum(tempo_vectors, axis = 0) #total number of word wj under the class/topic Ci
        NCm = np.sum(NwCm)#total number of words under the class
        PROB_WCm = []
        for i in range(len_dict):
            a = NwCm[i] + alpha
            b = NCm + alpha*len_dict
            PROB_WCm.append(a/b)
        VECTOR_PROBWCm.append(PROB_WCm)
    VECTOR_PROBWCm = np.array(VECTOR_PROBWCm, dtype= np.float64)

    test_vectors=np.array(test_df.loc[:,col_names],dtype=np.float64)
    temp_vectors = np.array(test_vectors>0,dtype = np.float64)
    for i in range(len(test_vectors)):
        PROB_CmDt = []
        test_vect = temp_vectors[i]
        for j in range(no_of_topics):
            train_prob = VECTOR_PROBWCm[j]
            prod_vect = (test_vect*train_prob)
            t = test_vect.copy()
            t[t == 0] = 1
            p = prod_vect.copy()
            p[ p == 0] =1
            PROB_DtCm = np.prod(p)
            PROB_CmDt.append(PROB_DtCm)
        maxP = 0
        com = 0
        for lm in range(no_of_topics):
            if PROB_CmDt[lm]>maxP:
                maxP = PROB_CmDt[lm]
                com = lm
        lb = 'a'
        if maxP == 0:
            lb = "Can't determine"
            test_df.loc[i,'result'] = lb
        elif maxP>0:
            lb = labels[com]
            test_df.loc[i, 'result'] = lb
        
    
    accuracy = FindAccuracy(test_df)
    return accuracy

def DistDfList(algo):
    dist_df_list = []
    for i in range(len(_test_vectors)):
            dist_list = []
            test_vect=_test_vectors[i]
            if algo == 'h':
                test_vect=np.array(test_vect>0,dtype=np.int8)
            elif algo == 'co':
                test_vect = VectorWithTFIDF(test_vect,_vectors)
            for j in range(len(_vectors)):
                train_vect=_vectors[j]
                if algo == 'h':
                    distance = MeasureHammingDistance(test_vect,train_vect)
                elif algo == 'e':
                    distance = MeasureEucledianDistance(test_vect,train_vect)
                else:
                    distance = MeasureCosineSimilarity(test_vect,train_vect)
                temp = []
                temp.append(distance)
                temp.append(df.loc[j,'labels'])
                dist_list.append(temp)
            dist_df=pd.DataFrame(dist_list)
            if algo == 'co':
                dist_df=dist_df.sort_values(0,ascending=False)
            else:
                dist_df=dist_df.sort_values(0)
            dist_df_list.append(dist_df)
    return dist_df_list

def ApplyHamming(train_dataframe,test_dataframe):
    dist_df_list = DistDfList('h')
    ACC = LoopForK(dist_df_list)
    return ACC

def ApplyEuclid(train_dataframe,test_dataframe):
    dist_df_list = DistDfList('e')
    ACC = LoopForK(dist_df_list)
    return ACC

def ApplyCosine(train_dataframe,test_dataframe):
    dist_df_list = DistDfList('co')
    ACC = LoopForK(dist_df_list)
    return ACC

def ApplyKNN(train_dataframe,test_dataframe):
    list_of_acc =[]
    ACCURACY = ApplyHamming(train_dataframe,test_dataframe)
    print('Hamming accuracy list for k=1,3,5:', ACCURACY)
    list_of_acc.append(ACCURACY)
    ACCURACY = ApplyEuclid(train_dataframe,test_dataframe)
    print('Euclid accuracy list for k=1,3,5:', ACCURACY)
    list_of_acc.append(ACCURACY)
    ACCURACY = ApplyCosine(train_dataframe,test_dataframe)
    print('Cosine accuracy list for k=1,3,5:', ACCURACY)
    list_of_acc.append(ACCURACY)
    return list_of_acc
############################################################COMMON CODE STARTS############################################################

labels = ExtractTopicNames("topics.txt")
train_topics_xml = ExtractTopicNames("topics.txt")
test_topics_xml = ExtractTopicNames("topics.txt")
print('Topic names:' , train_topics_xml)

for i in range(len(train_topics_xml)):
    train_topics_xml[i] = 'Training/' + str(train_topics_xml[i])+'.xml'

for i in range(len(test_topics_xml)):
    test_topics_xml[i] = 'Test/' + str(test_topics_xml[i])+'.xml'


stop_word_list = MakeStopWordDoc("Stopwords.txt")

index = 0
dictionary = {}
word_count=[]

max_rows = 200
for i in range(len(train_topics_xml)):
    DOMTree = xdm.parse(train_topics_xml[i])
    examples = DOMTree.getElementsByTagName("row")
    c = 0
    for row in examples:
        if c == (max_rows):
            break
        body = row.getAttribute("Body")
        body = body.lower()
        strings = re.findall(r'\w+',body)
        for word in strings:
            if len(strings)<5:
                continue
            if CheckStopWord(word)== 'YES':
                continue
            if word not in dictionary:
                dictionary[word] = index
                index = index+1
    
        c = c+1
#print(len(dictionary))
print("Dictionary is ready!")
col_names = list(range(0,len(dictionary)))
len_dict = len(dictionary)
no_of_topics = len(labels)

####### updating dataframe values by counting words
print('Making training dataframe...')
train_dataframe = MakeDataframe(max_rows,dictionary)
print('done')
print('Updating training dataframe...')
train_dataframe = UpdateDataframe(train_dataframe,max_rows,dictionary,train_topics_xml)
print('done')
#########testing starts here########


###CREATING TEST DATAFRAME
max_test_rows = 20
print('Making test dataframe...')
test_dataframe = MakeDataframe(max_test_rows,dictionary)
print('Updating test dataframe...')
test_dataframe = UpdateDataframe(test_dataframe,max_test_rows,dictionary,test_topics_xml)
print('done')

len_test_df = len(test_dataframe)

test_dataframe['result'] = 'no result yet' # adding a new column named result


df = train_dataframe.copy()
test_df = test_dataframe.copy()
_vectors=np.array(df.loc[:,col_names],dtype=np.float64)
_test_vectors=np.array(test_df.loc[:,col_names],dtype=np.float64)

k_values = [1,3,5]
D = max_rows*len(labels)
temp_vectors = np.array(_vectors>0,dtype = np.float64)
weight_vect = np.sum(temp_vectors,axis = 0)
############################################################COMMON CODE ENDS############################################################

############################################################Application#################################################################

f=open('Report2_NaiveBayes.txt','w')
f.write('Accuracy for 50 smoothing factors for Naive Bayes is given below:\n')
f.write('Iteration\t\tALPHA\t\t\t\t\tACCURACY\n')

sum_acc  = 0
accuracy = 0
count = 0
sm = np.linspace(.00005,.5,51)
for i in range(len(sm)-1):
    count = count +1
    acc = ApplyNaiveBayes(train_dataframe,test_dataframe,sm[i])
    sum_acc = sum_acc + acc
    f.write(str(count))
    f.write('\t\t')
    #print('smoothing factor:',sm[i+1])
    f.write(str(sm[i]))
    f.write('\t\t\t\t\t')
    f.write(str(acc))
    f.write('\n')
NB_AVG = sum_acc/50
f.close()


f=open('Report1_KNN.txt','w')
f.write('Accuracy for 3 values for 3 algorithms are given below:\n')
f.write('Algorithm\tK\tACCURACY\n')
ACCURACY = ApplyHamming(train_dataframe,test_dataframe)
for i in range(len(k_values)):
    f.write("Hamming")
    f.write('\t\t')
    f.write(str(k_values[i]))
    f.write('\t')    
    f.write(str(ACCURACY[i]))
    f.write('\n')
ACCURACY = ApplyEuclid(train_dataframe,test_dataframe)
for i in range(len(k_values)):
    f.write("Euclid")
    f.write('\t\t')
    f.write(str(k_values[i]))
    f.write('\t')    
    f.write(str(ACCURACY[i]))
    f.write('\n')
ACCURACY = ApplyCosine(train_dataframe,test_dataframe)
for i in range(len(k_values)):
    f.write("Cosine")
    f.write('\t\t')
    f.write(str(k_values[i]))
    f.write('\t')   
    f.write(str(ACCURACY[i]))
    f.write('\n')
f.close()

LIST_OF_ACC = ApplyKNN(train_dataframe,test_dataframe)
max_acc = 0
knn_algo = ['Hamming','Euclid','Cosine']
for i in range(len(knn_algo)):
    for j in range(len(k_values)):
        if LIST_OF_ACC[i][j]>max_acc:
            max_acc = LIST_OF_ACC[i][j]
            best_k = k_values[j]
            best_KNN = knn_algo[i]
f=open('BESTKNN_Vs_NB.txt','w')
f.write("Topic names:")
f.write(str(labels))
f.write('\n')
f.write("Best accuracy for best value of k for best KNN algorithm is:\n")
f.write("Best accuracy:")
f.write('\t')
f.write(str(max_acc))
f.write('\n')
f.write("Best KNN:")
f.write('\t')
f.write(str(best_KNN))
f.write('\n')
f.write("Best K:")
f.write('\t\t')
f.write(str(best_k))
f.write('\n')
f.write("Average accuracy for Naive Bayes: ")
f.write(str(NB_AVG))
f.close()   


print('max_rows:',max_rows)
print('max_test_rows:',max_test_rows)
print(test_df)
#CODE END
execution_time = time.time()-start_time
print('execution_time:' , execution_time)
    

