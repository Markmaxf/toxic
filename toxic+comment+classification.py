
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import csv
df=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
sample=pd.read_csv('sample_submission.csv')


# In[ ]:


df.comment_text[2098]


# In[ ]:


df.shape


# In[ ]:


df_test.shape


# In[ ]:


# fill na value with 'the reader here is not going', since no comment general mean not toxic, general
df_test.isnull().any()
df_test.comment_text.fillna(value='the reader here is not going',inplace=True)
df_test.isnull().any()


# In[ ]:


# text processing with nltk package 
# normal processing, not better
# turn 'kiss off' into 'kiss', which should be the main parameter decide the 'toxic' label 
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def re_process(dataset):
    process=[]
    for i in range(dataset.shape[0]):
        b=re.sub('\n',' ',dataset['comment_text'][i])
        b=re.sub('[0-9]',' ',b)
        b=b.lower()
        b=b.split()
        #c=[ps.stem(word) for word in b if not word in stopwords.words('english')]
        c=b
        process.append(' '.join(c))
    return process
train_re=re_process(df)
test_re=re_process(df_test)


# In[ ]:


# text processing with nltk package 
# should keep stops words or delete self-define stropwords 
train_re


# In[ ]:


from keras.preprocessing import text


# In[ ]:


# text processing with keras text processing package 
# for lstm model 
# better? like 'toxic' label of comment 1 should be heavily weight by kiss off
def keras_process(dataset):
    process=[]
    for i in range(dataset.shape[0]):
        a=text.text_to_word_sequence(dataset['comment_text'][i])
        process.append(' '.join(a))
    return process
train_keras=keras_process(df)
test_keras=keras_process(df_test)


# In[ ]:


#remove the stopwords
# since words 'kiss off' is totally different with 'kiss', so only remove part of stops words
def cut_word(in_word):
    out_word=in_word
    if len(in_word)>30:
        out_word='invalid'
    return out_word
def remove_word(data):
    process=[]
    for i in range(len(data)):
        if i%10000==0:
            print(i)
        b=data[i]
        b1=b.split()
        if len(b1)>1000:
            b1=b1[:1000]
        c=[ps.stem(cut_word(word)) for word in b1 if not word in stopwords.words('english')[:80]]
        process.append(' '.join(c))
    return process


# In[ ]:


train_keras_clean=remove_word(train_keras)


# In[ ]:


test_keras_clean=remove_word(test_keras)


# In[ ]:


train_re_in=pd.read_csv('train_processed_with_re.csv')
test_re_in=pd.read_csv('test_processed_with_re.csv')
train_re_in.fillna(value='day month year',inplace=True)
test_re_in.fillna(value='day month year',inplace=True)
train_re=np.array(train_re_in.comment_text)
test_re=np.array(test_re_in.comment_text)


# In[ ]:


# calculate the length of text for sequence analysis
def cal_len(data):
    lenall=[]
    for i in range(len(data)):
        #print(i)
        k=data[i].split()
        lenall.append(len(k))
    return lenall
train_keras_clean_len=cal_len(train_re)
test_keras_clean_len=cal_len(test_re)


# In[ ]:


plt.subplot(211)
plt.hist(train_keras_clean_len,bins=50)
plt.subplot(212)
plt.hist(test_keras_clean_len,bins=50)
plt.xlabel('length of review')


# In[ ]:


# Vectorize the text,
#max_featur_tfidf=3000

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
#raw=np.hstack([np.array(train_keras_clean),np.array(test_keras_clean)])
tf.fit(train_re)


# In[ ]:


train_keras_data=tf.transform(train_re)
test_keras_data=tf.transform(test_re)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
y=df.iloc[:,2:8].values
x_train,x_test,y_train,y_test=train_test_split(train_keras_data,y,test_size=0.1,random_state=1)


# In[ ]:


# train vectorized data with Randomforest model 
Rc=RandomForestClassifier()
Rc.fit(x_train,y_train)
test_pre=Rc.predict(test_keras_data)


# save results to csv
sample=pd.read_csv('sample_submission.csv')


wirte=np.hstack([np.array(df_test['id']).reshape(-1,1),test_pre])

#out=pd.DataFrame(wirte,columns=['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate'])

out1=pd.DataFrame(test_pre,columns=sample.columns[1:8])
out1.head()
out1['id']=sample.id.astype(str)
out1.head()

out1.to_csv('results_randome_with_punctuation.csv',index=False,sep=',')


# In[ ]:


# training vectorized data with Logistic and Bayes model 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB

LR=LogisticRegression()
output=np.ones([153164,1])
for i in range(6):

    LR.fit(x_train,y_train[:,i])
    LR_pre=LR.predict(test_keras_data).reshape([153164,1])
    output=np.hstack([output,LR_pre])   
#NB=GaussianNB()
#NB.fit(x_train,y_train)


# In[ ]:


# predict text data

test_pre=Rc.predict(test_keras_data)

# evaluate with AUC metricx
from sklearn.metrics import classification_report
from sklearn.metrics import auc
from sklearn.metrics import log_loss

log_loss(y_test,y_pre)


# In[ ]:


# save the results
sample=pd.read_csv('sample_submission.csv')



out1=pd.DataFrame(test_pre,columns=df.columns[2:8])

out1['id']=df_test.id

wirte=np.hstack([np.array(df_test['id']).reshape(-1,1),test_pre])

out=pd.DataFrame(wirte,columns=['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate'])

out1=pd.DataFrame(test_pre,columns=sample.columns[1:8])
out1.head()
out1['id']=sample.id.astype(str)
out1.head()

out1.to_csv('results_logistic.csv',index=False,sep=',')


# In[ ]:


sample.columns[1:8]

out2=pd.DataFrame({'id':sample.id.astype(str),
                   'identity_hate':test_pre[:,5],
                   'insult':test_pre[:,4],
                   'threat':test_pre[:,3],
                  'obscene':test_pre[:,2],
                  'severe_toxic':test_pre[:,1],
                  'toxic':test_pre[:,0]
                   
                   
                
                
#                  })

#out2.to_csv('results2.csv',index=False,sep=',')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# In[ ]:


# training data with LSTM model
import keras.preprocessing.text as text 
Token=text.Tokenizer(num_words=50000)
Token.fit_on_texts(train_re)


# In[ ]:


train_word_sequence=Token.texts_to_sequences(train_re)
test_word_sequence=Token.texts_to_sequences(test_re)


# In[ ]:


from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input,Embedding,GRU,Dropout,Dense,concatenate,LSTM

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Bidirectional,GlobalAveragePooling1D

max_len_sequence=80

train_x=pad_sequences(train_word_sequence,maxlen=max_len_sequence)
test_x=pad_sequences(test_word_sequence,maxlen=max_len_sequence)


max_text=np.max([np.max(train_x.max()),np.max(test_x.max())])+1
#max_text=50000
#max_text=np.max([np.max(np.array(train_word_sequence).reshape(-1,1).max()),
#                np.max(np.array(test_word_sequence).reshape(-1,1).max())])+1
comment_keras=Input(shape=(max_len_sequence,))
comment_embed=Embedding(input_dim=max_text, output_dim=30)(comment_keras)

lr=0.1
comment_lstm=Bidirectional(LSTM(64,activation='relu'))(comment_embed)
model1=Dense(32,activation='sigmoid')(comment_lstm)
model2=Dropout(lr)(model1)


output=Dense(6,activation='softmax')(model2)

model=Model(inputs=comment_keras,outputs=output)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:


# used for tuning the parameter
del model,comment_keras,comment_embed,comment_lstm


# In[ ]:


max_text=np.max([np.max(np.array(train_word_sequence).reshape(-1,1).max()),
                np.max(np.array(test_word_sequence).reshape(-1,1).max())])+1
max_text


# In[ ]:


# training and save the predicted results from text resutls
y=df.iloc[:,2:8].values
model.fit(train_x,y,batch_size=10000,epochs=30,validation_split=0.15)

test_pre=model.predict(test_x)
out2=pd.DataFrame({'id':sample.id.astype(str),
                   'identity_hate':test_pre[:,5],
                   'insult':test_pre[:,4],
                   'threat':test_pre[:,3],
                   'obscene':test_pre[:,2],
                   'severe_toxic':test_pre[:,1],
                   'toxic':test_pre[:,0]
                   
                   
                
                
                  })

out2.to_csv('sample_lstm_30.csv',sep=',',index=False,encoding='utf-8')


# In[ ]:


out2.head()


# In[ ]:


# randomly check the data,1002
train_re[1004]


# In[ ]:


# check the data 
df.comment_text[1004]


# In[ ]:


del model 


# In[ ]:


len(train_word_sequence)


# In[ ]:


out2.head()


# In[ ]:


lstm=pd.read_csv('sample_lstm_2.csv')


# In[ ]:


lstm.head()


# In[ ]:


# ensemble the data of different training model
tfidf=pd.read_csv('results_tfidf.csv')

loigstic=pd.read_csv('results_logistic.csv')

c=pd.concat((tfidf,lstm,loigstic),axis=0).groupby('id').mean()


# In[ ]:


c.to_csv('lstm_and_tfidf_logistic_ensemble.csv')


# In[ ]:




