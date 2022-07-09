#!/usr/bin/env python
# coding: utf-8

# Importing the required Libraries
import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx

# Extract word vectors
word_embeddings = {}
file = open(r'..\\NLP_Project\\glove.6B.100d.txt', encoding='utf-8')
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
file.close()
len(word_embeddings)


df = pd.read_excel('..\\NLP_Project\\TASK.xlsx')

df.columns


#renaming the cloumn
df.rename(columns = {'Unnamed: 1' : 'Introduction' }, inplace=True)
# Deleting the first row
df.drop(0)


# Converting the DataFrame into a dictionary
text_dictionary = {}
for i in range(1,len(df['TEST DATASET'])):
    text_dictionary[i] = df['Introduction'][i]
    
print(text_dictionary[1])


# function to remove stopwords
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# function to make vectors out of the sentences
def sentence_vector_func (sentences_cleaned) : 
    sentence_vector = []
    for i in sentences_cleaned:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vector.append(v)
    
    return (sentence_vector)


# function to get the summary of the articles
def summary_text (test_text, n = 5):
    sentences = []
    
    # tokenising the text 
    sentences.append(sent_tokenize(test_text))
    # print(sentences)
    sentences = [y for x in sentences for y in x] # flatten list
    # print(sentences)
    
    # remove punctuations and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-z A-Z 0-9]", " ",regex=True)

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    #print(clean_sentences)

    
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    #print(clean_sentences)
    
    sentence_vectors = sentence_vector_func(clean_sentences)
    
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    #print(sim_mat)
    
    # Finding the similarities between the sentences 
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    
    
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    #print(scores)
    
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)))
    # Extract sentences as the summary
    summarised_string = ''
    for i in range(n):
        
        try:
            summarised_string = summarised_string + str(ranked_sentences[i][1])            
        except IndexError:
            print ("Summary Not Available")
    
    return (summarised_string)


print("Kindly let me know in how many sentences you want the summary - ")
x = int(input())

summary_dictionary = {}

for key in text_dictionary:
    
    para = text_dictionary[key]
    print("Summary of the article - ",key)
    summary = summary_text(para,x)
    summary_dictionary[key] = summary
    
    print(summary)   
    
print ("************* The process has been completed successfully *************")


summary_table = pd.DataFrame(list(summary_dictionary.items()),columns = ['TEST DATASET','Summary'])

data_table = pd.DataFrame(list(text_dictionary.items()),columns = ['TEST DATASET','Introduction'])

# Combining the findings into the table
result  = pd.concat([data_table , summary_table['Summary']], axis = 1 , sort = False)
print(result)


result.to_csv("Summary_File.csv")

