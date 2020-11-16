#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdb
import pickle
import string
import numpy as np
import pandas as pd

import time

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy
import sklearn
from gensim.models import KeyedVectors
from nltk.corpus import stopwords, twitter_samples
from nltk.tokenize import TweetTokenizer

from utils import (cosine_similarity, get_dict,
                   process_tweet)


# In[2]:


import nltk
from gensim.models import KeyedVectors

data = pd.read_csv('HS_sep19.csv')
df = pd.DataFrame()
df['desc'] = data['DESC']
dfl = df.values.tolist()

embeddings = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True)


# In[3]:


f = open('desc.csv', 'r').read()
set_words = set(nltk.word_tokenize(f))

def get_word_embeddings(embeddings):

    word_embeddings = {}
    for word in embeddings.vocab:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings


# Testing your function
word_embeddings = get_word_embeddings(embeddings)
print(len(word_embeddings))
pickle.dump( word_embeddings, open( "word_embeddings_subset.p", "wb" ) )


# In[4]:


set_words


# In[5]:


word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))
len(word_embeddings) 


# In[6]:


tea = word_embeddings['tea']
coffee = word_embeddings['coffee']

cosine_similarity(tea, coffee)


# In[7]:


def get_document_embedding(tweet, en_embeddings): 
    '''
    Input:
        - tweet: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - tweet_embedding: a
    '''
    doc_embedding = np.zeros(300)

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # process the document into a list of words (process the tweet)
    processed_doc = process_tweet(tweet)
    for word in processed_doc:
        # add the word embedding to the running total for the document embedding
        doc_embedding+=en_embeddings.get(word,0)
    ### END CODE HERE ###
    return doc_embedding


# In[8]:


custom_tweet = data.loc[5,'DESC']
print(process_tweet(custom_tweet))
tweet_embedding = get_document_embedding(custom_tweet, word_embeddings)
tweet_embedding[-5:]


# In[9]:


df = df.reset_index()
dfl = df.values.tolist()
dfl[5]


# In[10]:


def get_document_vecs(all_docs, en_embeddings):
    '''
    Input:
        - all_docs: list of strings - all tweets in our dataset.
        - en_embeddings: dictionary with words as the keys and their embeddings as the values.
    Output:
        - document_vec_matrix: matrix of tweet embeddings.
        - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
    '''

    # the dictionary's key is an index (integer) that identifies a specific tweet
    # the value is the document embedding for that document
    ind2Doc_dict = {}

    # this is list that will store the document vectors
    document_vec_l = []

    for i, doc in all_docs:

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # get the document embedding of the tweet
        doc_embedding = get_document_embedding(doc,en_embeddings)

        # save the document embedding into the ind2Tweet dictionary at index i
        ind2Doc_dict[i] = doc_embedding

        # append the document embedding to the list of document vectors
        document_vec_l.append(doc_embedding)

        ### END CODE HERE ###

    # convert the list of document vectors into a 2D array (each row is a document vector)
    document_vec_matrix = np.vstack(document_vec_l)

    return document_vec_matrix, ind2Doc_dict


# In[11]:


document_vecs, ind2Tweet = get_document_vecs(dfl, word_embeddings)


# In[12]:


print(f"length of dictionary {len(ind2Tweet)}")
print(f"shape of document_vecs {document_vecs.shape}")


# In[14]:


dic_embedding = {}
for i in range(0,len(data)):
        mem = data.loc[i,"DESC"]
        mem_embedding = get_document_embedding(mem, word_embeddings)
        dic_embedding[i] = mem_embedding


# In[15]:


data_final = pd.DataFrame()
from collections import Counter 
from textblob import TextBlob
df_check = pd.read_csv('test.csv')
test = pd.DataFrame()
test['desc'] = df_check['desc']


for j in test['desc']:
    my_tweet = j
    my_tweet = my_tweet.lower() 
    b = TextBlob(my_tweet)
    my_tweet = str(b.correct())
    #print(my_tweet)
    #print(process_tweet(my_tweet))

    tweet_embedding = get_document_embedding(my_tweet, word_embeddings)
    
    dic = {}
    for i in range(0,len(data)):
        temp_embedding = dic_embedding[i]
        #print(cos_sim)
        #print(temp)
        dic[i] = cosine_similarity(tweet_embedding, temp_embedding)
        """if cos_sim >= c:
            c = cos_sim
            d = temp
            hsc = data.loc[i,"HS_Code"]"""


    k = Counter(dic)
    high = k.most_common(1) 
    #print(high)
    #dic = sorted(dic.items(),key = lambda x:x[1],reverse = True)
    #print(d)
    #print(process_tweet(d))
    #print(c)
    #print(hsc)
    for i in high:
        data_final['input'] = my_tweet
        data_final['index_at'] = i[0]
        data_final['score'] = i[1]
        data_final['matched_with'] = data.loc[i[0],"DESC"]
        #print(i[0]," :",i[1]," ") 
        #print(data.loc[i[0],"DESC"])


# In[ ]:


data_final.to_csv(r'C:\Users\Palkit Lohia\Desktop\data_final.csv', index = False, header=True)


# In[ ]:


"""my_tweet = "CAPS THIS SHIPMENT CONTAINS NO SOLID WOOD PACKING MATERIALS. . . ."
my_tweet = my_tweet.lower()
from textblob import TextBlob 
b = TextBlob(my_tweet)
 
# prints the corrected spelling
my_tweet = str(b.correct())


print(my_tweet)
print(process_tweet(my_tweet))

tweet_embedding = get_document_embedding(my_tweet, word_embeddings)"""


# In[ ]:


"""from collections import Counter 
#c = -10
dic = {}
for i in range(0,len(data)):
    temp = data.loc[i,"DESC"]
    temp_embedding = get_document_embedding(temp, word_embeddings)
    #print(cos_sim)
    #print(temp)
    dic[i] = cosine_similarity(tweet_embedding, temp_embedding)
    """
    """if cos_sim >= c:
            c = cos_sim
            d = temp
            hsc = data.loc[i,"HS_Code"]"""
        
        
"""k = Counter(dic)
high = k.most_common(1) 
#print(high)
#dic = sorted(dic.items(),key = lambda x:x[1],reverse = True)
#print(d)
#print(process_tweet(d))
#print(c)
#print(hsc)
for i in high: 
    print(i[0]," :",i[1]," ") 
    print(data.loc[i[0],"DESC"])"""


# In[ ]:


t = "men car"
t_embedding = get_document_embedding(t, word_embeddings)
q = "men bike"
q_embedding = get_document_embedding(q, word_embeddings)


# In[ ]:


cos_sim_t = cosine_similarity(q_embedding, t_embedding)


# In[ ]:


print(cos_sim_t)


# In[ ]:


#stemming and num-check
#spell-check
#top 5
#append new hs_code docs
#algo alternate to cosine similarity
#https://medium.com/@adriensieg/text-similarities-da019229c894


# In[ ]:


#excluding
#send_code
##uniform scoring for all tweets


# In[ ]:


#run the code for all 20k tweets and store the topmost result(hs_code, score and desc)
#perform normal statistics on the output
#uniform scoring for all tweets


# In[ ]:


#pre-process 3rd cell
#remove stemming from tweets
#spell-check after stemming
#storing tweet-embedding for the 6k desc in a dict - done

