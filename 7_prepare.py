from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import csv
import numpy as np
import os
import pandas as pd
import random

#1 moment
moment_file = open('./moment.txt', 'r')
moment = moment_file.read().split('\n')[:-1]
moment_file.close()
word_count=512

#l1 agency - yes no
#l2 social - yes no
processed=[]
data=[]
#data_f=open('./processed/labeled_a_n.csv','r')
#data_f=open('./processed/labeled_s_n.csv','r')
#data_f=open("./processed/labeled_test_data.csv",'r')
#data_f=open("./processed/unlabeled_data.csv",'r')
count=0
with open("./test/unlabeled_17k.csv", encoding='mac_roman', newline='') as csvfile:
    reader=csv.reader(csvfile,delimiter=',')
    count=0
    for row in reader:
        if count==0:
            count+=1
            continue
        #print(row)
        item=[]

        #hmid
        item.append(row[0])
        #moment
        count1=[0]*word_count
        #process sentence
        tokens=word_tokenize(row[1])
        tokens=[w.lower() for w in tokens]
        #remove punctuation
        table=str.maketrans('','',string.punctuation)
        stripped=[w.translate(table) for w in tokens]
        #remove not alphabetic tokens
        words=[word for word in stripped if word.isalpha()]
        #filter out stop words
        stop_words=set(stopwords.words('english'))
        words=[w for w in words if not w in stop_words]
        for word in words:
            #print(word)
            if word in moment:
                count1[moment.index(word)]=1
        for i in range(0,word_count):
            item.append(str(count1[i]))
        data.append(item)

p_file=open("./test/test_oh.csv", 'w')
print(len(data[0]))
#input train test
wr = csv.writer(p_file, dialect='excel')
for x in data:
    wr.writerow(x)
p_file.close()
