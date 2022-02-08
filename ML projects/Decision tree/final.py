import pandas as pd
import csv                     
import numpy as np
import sys
import math
sys.__stdout__=sys.stdout                        ## for printing df



# functionS for preparing dataframe from dataset file

def make_LoL(file_name):
    f = open(file_name)
    row_list=csv.reader(f,delimiter='\t')
    LIST = []
    initial_list = []
    for line in row_list:                    
        for i in range(len(line)):
            var = line[i]                        ## converting each line into string
            initial_list= var.split(',')         ## putting each element in a list
            LIST.append(initial_list)            ## putting each list in another list
    return LIST



def make_df(file_name):
    LIST = make_LoL(file_name)
    data = []
    data = LIST[1:]                               ## preparing list for using in dataframe by seprating data from attribute names
    df = pd.DataFrame(data,columns = LIST[0])
    return df

#find input attributes from current dataframe
def input_atts(data_frame):
    #in_atts = []
    #temp_df = data_frame.drop(target_attribute, axis=1)
    in_atts = data_frame.columns.values.tolist()      
    in_atts = in_atts[0:-1]
    return in_atts

# finding possible labels in an attribute
def find_labels(dframe,name_of_attribute):
    labels = []
    list_of_labels = dframe[name_of_attribute]
    for i in list_of_labels:
        if i not in labels:
            labels.append(i)
    return labels    



# list of count of each lables in an attribute
def count_of_label(dframe,name_of_attribute):
    count = []
    labels = find_labels(dframe,name_of_attribute)
    for i in labels:
        var = i
        c = 0
        list_of_labels = dframe[name_of_attribute]
        for m in list_of_labels:
            if m == var:
                c = c+1
        count.append(c)
    return count

# finding entropy of an attribute
def entropy(dframe,name_of_attribute,n_data):
    labels = find_labels(dframe,name_of_attribute)
    count = count_of_label(dframe,name_of_attribute)
    s = 0
    p = 0
    for i in range(len(labels)):
        p = (count[i])/ n_data
        s = s- p*(math.log2(p))
    return s

def info_gain(dframe,name_of_attribute,n_data):
    SUM = 0
    labels = find_labels(dframe,name_of_attribute)
    count_for_labels = count_of_label(dframe,name_of_attribute)
    grouped=dframe.groupby(name_of_attribute)
    for i in range(len(labels)):
        temp_df=(grouped.get_group(labels[i]))
        temp_en = entropy(temp_df,target_attribute,count_for_labels[i])
        term = ((count_for_labels[i])/n_data)*temp_en
        SUM = SUM + term
    total_en = entropy(dframe,target_attribute,n_data)
    sv = total_en - SUM
    return sv



def find_best_att(dframe,input_attributes): 
    maxG=0
    best = ''
    for i in range(len(input_attributes)):                                       
        g= info_gain(dframe,input_attributes[i],n_data)
        if g>maxG:
            maxG=g
            best=input_attributes[i]
    return best


result=[]
p=0
root=''
summ=[]
final=[]
temp=[]
def ID3(dframe,target_attribute,input_attributes):
    global temp, p, root, res, final
    p = p+1
    check_label = find_labels(dframe,target_attribute)
    if len(check_label)== 1:                                    ##check if target att has same values
        return check_label[0]
    labels_of_t_att = []
    count_of_t_att_labels = []
    labels_of_t_att =  find_labels(dframe,target_attribute)
    count_of_t_att_labels = count_of_label(dframe,target_attribute)
    mc = 0
    save_i = 0
    for i in range(len(count_of_t_att_labels)):
        m = count_of_t_att_labels[i]
        if m>mc:
            mc = m
            save_i = i
    most_common = labels_of_t_att[save_i]
    
    if len(input_attributes)==0:                 
        return most_common
    new_n_data = len(dframe.index)                              ##calculates number of remaining data(rows) in current data frame
    total_entropy = entropy(dframe,target_attribute,new_n_data) ##calculates total entropy of new dataframe
    
    Best_att = find_best_att(dframe,input_attributes)
    
    if p ==1:
        root = Best_att                                         ##saves 1st best attribute as root

    labels_of_best_att = find_labels(dframe,Best_att)
    count_of_best_att_labels = count_of_label(dframe,Best_att)
    grouped=dframe.groupby(Best_att)
    for i in labels_of_best_att:
        new_df = grouped.get_group(i)
        new_df = new_df.drop(Best_att, axis=1)               ## dropping current best attribute
        no_of_current_column = len(new_df.columns.values)    ## counts number of existing columns(attributes) in dataframe
        temp.append(Best_att)                                
        temp.append(i)
        #check if all input atts are dropped
        if no_of_current_column == 1:                        ## means only target attribute is left in df
            '''labels_of_t_att = []
            count_of_t_att_labels = []
            labels_of_t_att =  find_labels(new_df,target_attribute)
            count_of_t_att_labels = count_of_label(new_df,target_attribute)
            mc = 0
            save_i = 0
            for i in range(len(count_of_t_att_labels)):
                m = count_of_t_att_labels[i]
                if m>mc:
                    mc = m
                    save_i = i
            most_common = labels_of_t_att[save_i]'''
            return most_common
        else: 
            new_input_attributes = input_atts(new_df)
            
            ret = ID3(new_df,target_attribute,new_input_attributes)
            temp.append(target_name)
            temp.append(ret)
            
            final.append(temp)                                  ####creating list of list 
            temp=[]                                             ####emptying current rule
    return

def find_tree(final,root):
    root=final[0][0]
    for m in range(len(final)):
        if final[m][0]==root:
            tempo=final[m]
        if final[m][0]!=root:
            for t in range(len(tempo)):
                if tempo[t]==final[m][0]:
                    tempo=tempo[:t]
                    break
            final[m]=tempo+final[m]
        #print(final[m])                                        #### print each rule

def test_accuracy(test_dataframe,t_att_test,final):
    c=0
    for s in range(len(test_dataframe)):
         test=test_dataframe.loc[s,:]
         
         for i in range(len(final)):
            m=len(final[i])
            for j in range(int(m/2)):
                attr=final[i][2*j]  
                lb=final[i][2*j+1]
                
                if attr!=target_attribute:
                   if test[attr] != lb:
                        break
            if attr == target_attribute:
                res=lb
                
                if t_att_test[s]==res:
                    c=c+1
                break
            if attr==target_attribute:       
                break
    acc=c/len(test_dataframe)*100
    print('No of accurate prediction:',c)
    print('total data in dataframe :',len(test_init_dframe))
    print('accuracy :',c/len(test_dataframe)*100,'%')
    
    return acc
#########calling

###############phase:training################
file_name = 'car.csv'                         ###### input 
LIST = make_LoL(file_name)                    ##contains totale file as list

no_of_row = len(LIST)                         ## row in entire list
no_of_column = len(LIST[0])                   ## row in entire column
n_data = no_of_row-1                          ## assuming first row is attribute name, 

attribute_names = LIST[0]                     ## contains all attribute names of dataset 
target_attribute = LIST[0][-1]                ## assuming last attribute is target attribute
target_name = target_attribute
init_dframe = make_df(file_name)              ## data frame of given file
init_dframe = init_dframe.sample(frac=1)
#init_dframe = init_dframe.drop('day',axis=1)  ##for droping an unnecessary attribute
#input_attributes = input_atts(init_dframe)    ## total given input attributes
target_attr = init_dframe[target_attribute]

index=int(0.8*len(init_dframe))
start=index
end=index
#start=0
#end=len(init_dframe)
test_init_dframe=init_dframe.loc[start:,:]
target_label_test=test_init_dframe[target_name]
test_df=test_init_dframe.reset_index(drop=True)
target_label_test=target_label_test.reset_index(drop=True)
init_dframe=init_dframe.loc[:end,:]
input_attributes = input_atts(init_dframe)
target_attr=target_attr.loc[:end]





ID3(init_dframe,target_attribute,input_attributes)
find_tree(final,root)
print(len(final))                               #prints no of rule
accuracy=test_accuracy(test_df,target_label_test,final)


