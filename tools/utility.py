# IMPORTS
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from copy import deepcopy
import ast
from scipy import stats
import re
import matplotlib.pyplot as plt


import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
stop_words = stopwords.words('french')

from sklearn.feature_extraction.text import TfidfVectorizer 



##### Tokenize corpus inplace #####
def tokenize_corpus(corpus):   
    for document in tqdm(corpus):
        plain_text = document["corpus"]
        plain_text = plain_text.lower()
        plain_text= re.sub(r'\s+', ' ', plain_text)
        #plain_text = re.sub("[^a-z0-9]", ' ', plain_text)
        plain_text = re.sub("[^a-z]", ' ', plain_text)
        plain_text = re.sub(r'\s+', ' ', plain_text)
        #remove one letter words?
        #remove numbers?
        pt_words = word_tokenize(plain_text)
        cleaned_words =list()
        for word in pt_words:
            if len(word)>1:
                if word not in stop_words:
                    cleaned_words.append(word)
        document["corpus"] = cleaned_words
    return corpus
  
    
##### clean_corpus_labels #####
def clean_corpus_labels (corpus, siren_filtered):
    removed = 0
    for document in corpus:
        sirens = list()
        for siren in document["siren"]:
            if siren in siren_filtered:
                sirens.append(siren)
            else:
                removed+=1
        document["siren"]=sirens
    print ("Number of labels removed:",removed)
    return corpus


##### Functions for Fetching relevant words using Tf.Idf on Company related articles #####

def identity_tokenizer(text):
    return text

# generate relevant words using TF-IDF (sublinear)
def generate_relevant_words_tfidf(corpus,list_siren):
    relevant_words_tfidf = {}
    for siren in tqdm(list_siren):
        plain_text_list = list()
        company_article = list()
        #binary = True
        #sublinear_tf=False
        tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range = (1,1), lowercase=False, sublinear_tf=True)
        #tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range = (1,1), lowercase=False, sublinear_tf=False)
        #Building "forground"
        for document in corpus:
            if siren in document["siren"]:
                company_article = company_article+document["corpus"]  # add article to company BIG article
            else:
                plain_text_list.append(document["corpus"]) # otherwise add to corpus

        plain_text_list.insert(0,company_article) # add company article to begging of corpus
        tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(plain_text_list)

        #Get the tf-idf scores for the words in the company article complication.(=forground)
        first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0] # discard tf.idf scores for the other texts

        # place tf-idf values in a pandas data frame 
        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"]) 
        df = df.sort_values(by=["tfidf"],ascending=False).head(40) # Take top 40 words

        relevant_words_tfidf[siren] = list(zip(list(df.index),list(df["tfidf"]))) # format result
    return relevant_words_tfidf

#####