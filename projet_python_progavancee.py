#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/05/2020

@author: elliot et mehdi
"""

################################## Déclaration des classes ##################################

import datetime as dt
from sklearn import *
import pickle
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

class Corpus():
    
    def __init__(self,name):
        self.name = name
        self.collection = {}
        self.authors = {}
        self.id2doc = {}
        self.id2aut = {}
        self.ndoc = 0
        self.naut = 0
            
    def add_doc(self, doc):
        
        self.collection[self.ndoc] = doc
        self.id2doc[self.ndoc] = doc.get_title()
        self.ndoc += 1
        aut_name = doc.get_author()
        aut = self.get_aut2id(aut_name)
        if aut is not None:
            self.authors[aut].add(doc)
        else:
            self.add_aut(aut_name,doc)
            
    def add_aut(self, aut_name,doc):
        
        aut_temp = Author(aut_name)
        aut_temp.add(doc)
        
        self.authors[self.naut] = aut_temp
        self.id2aut[self.naut] = aut_name
        
        self.naut += 1

    def get_aut2id(self, author_name):
        aut2id = {v: k for k, v in self.id2aut.items()}
        heidi = aut2id.get(author_name)
        return heidi

    def get_doc(self, i):
        return self.collection[i]
    
    def get_coll(self):
        return self.collection

    def __str__(self):
        return "Corpus: " + self.name + ", Number of docs: "+ str(self.ndoc)+ ", Number of authors: "+ str(self.naut)
    
    def __repr__(self):
        return self.name

    def sort_title(self,nreturn=None):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_title())][:(nreturn)]

    def sort_date(self,nreturn):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_date(), reverse=True)][:(nreturn)]
    
    def save(self,file):
            pickle.dump(self, open(file, "wb" ))
        
       

class Author():
    def __init__(self,name):
        self.name = name
        self.production = {}
        self.ndoc = 0
        
    def add(self, doc):     
        self.production[self.ndoc] = doc
        self.ndoc += 1

    def __str__(self):
        return "Auteur: " + self.name + ", Number of docs: "+ str(self.ndoc)
    def __repr__(self):
        return self.name
    


class Document():
    
    # constructor
    def __init__(self, date, title, author, text, url):
        self.date = date
        self.title = title
        self.author = author
        self.text = text
        self.url = url
    
    # getters
    
    def get_author(self):
        return self.author

    def get_title(self):
        return self.title
    
    def get_date(self):
        return self.date
    
    def get_source(self):
        return self.source
        
    def get_text(self):
        return self.text

    def __str__(self):
        return "Document " + self.getType() + " : " + self.title
    
    def __repr__(self):
        return self.title
    
    def getType(self):
        pass
    





########################################################################################
################################## Création du Corpus ##################################
########################################################################################
import praw
import urllib.request
import xmltodict

# On crée le corpus et on instancie des liste pour l'analyse des corpus dans le futur
corpus = Corpus("Corona")
corp_reddit = []
corp_arx = []

# On récupère les articles de Reddit
reddit = praw.Reddit(client_id='l8NSGqBrmubj_g', client_secret='0mUfKDkKNbDsrC1itDJJ5zqQ3sw4-Q', user_agent='Projet_Scrapping')
hot_posts = reddit.subreddit('Coronavirus').hot(limit=100)
for post in hot_posts:
    datet = dt.datetime.fromtimestamp(post.created)
    txt = post.title + ". "+ post.selftext
    txt = txt.replace('\n', ' ')
    txt = txt.replace('\r', ' ')
    corp_reddit.append(txt)
    doc = Document(datet,
                   post.title,
                   post.author_fullname,
                   txt,
                   post.url)
    corpus.add_doc(doc)
    
# On récupère les articles d'Arxiv
url = 'http://export.arxiv.org/api/query?search_query=all:covid&start=0&max_results=100'
data =  urllib.request.urlopen(url).read().decode()
docs = xmltodict.parse(data)['feed']['entry']

for i in docs:
    datet = dt.datetime.strptime(i['published'], '%Y-%m-%dT%H:%M:%SZ')
    try:
        author = [aut['name'] for aut in i['author']][0]
    except:
        author = i['author']['name']
    txt = i['title']+ ". " + i['summary']
    txt = txt.replace('\n', ' ')
    txt = txt.replace('\r', ' ')
    corp_arx.append(txt)
    doc = Document(datet,
                   i['title'],
                   author,
                   txt,
                   i['id']
                   )
    corpus.add_doc(doc)


########################################################################################
################################### TF - IDF METHOD ####################################
########################################################################################
# On vérifie de bien avoir les stopwords et punkt 
nltk.download('stopwords')
nltk.download('punkt')

# On découpe les corpus
corp_reddit = ','.join(corp_reddit)
corp_arx = ','.join(corp_arx)

# On instancie nos stopwords
en_stop = set(stopwords.words('english'))

# On "tokkenize" les corpus (création des dictionnaires de mots uniques)
reddit_tokens = word_tokenize(corp_reddit)
arx_tokens = word_tokenize(corp_arx)

# On compte la fréquence d'apparition des mots dans les corpus
reddit_stopped_tokens = [i for i in reddit_tokens if not i in en_stop]
arx_stopped_tokens = [i for i in arx_tokens if not i in en_stop]


reddit_stopped_tokens = ','.join(reddit_stopped_tokens)
arx_stopped_tokens = ','.join(arx_stopped_tokens)
corp_idf = [reddit_stopped_tokens, arx_stopped_tokens]


# On instancie la méthode TF-IDF
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corp_idf)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

# On en récupère les scores et les valeurs des mots pour ces scores les plus élevés
df2_words = df.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=25)
df2_scores = df.apply(lambda s, n: pd.Series(s.nlargest(n)), axis=1, n=25)


reddit_words = df2_words.iloc[0,]
reddit_words.columns = ['Word TF-IDF Score']
arx_words = df2_words.iloc[1,] 
arx_words.columns = ['Word TF-IDF Score']

reddit_scores = df2_scores.iloc[0,] 
arx_scores = df2_scores.iloc[1,] 

# On enlève les valeurs nulles et on trie par score décroissant.
reddit_scores = reddit_scores.sort_values(ascending=False)
reddit_scores = reddit_scores.dropna()
arx_scores = arx_scores.sort_values(ascending=False)
arx_scores = arx_scores.dropna()

# On modèlise nos scores dans des dataframes respectifs
reddit_scores = pd.DataFrame(reddit_scores)
arx_scores = pd.DataFrame(arx_scores)


###########################################################################################
################################### SIMILARITE COSINUS ####################################
###########################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#redarx1 contient tout le corpus reddit concaténer en 1 chaine 
#redarx2 pareil mais pour arxiv
redarx1 = ', '.join(corp_reddit)
redarx2 = ', '.join(corp_arx)

#redarx contient tout le corpus
redarx = [reddit_stopped_tokens, arx_stopped_tokens]

# Initialisation des vecteurs afin d'obtenir la similarité cosinus
count_vectorizer = TfidfVectorizer(stop_words='english')


# On crée la matrice des poids des mots afin de dégager ceux qui sont le plus importants
# redarx contient 2 chaines de caracteres que la fonction va comparer, le corpus reddit et celui arxiv
sparse_matrix = count_vectorizer.fit_transform(redarx)

# On affiche la Similarité Cosinus
aa = pd.DataFrame(cosine_similarity(sparse_matrix))
#print(cosine_similarity(sparse_matrix))


###########################################################################################
############################ ANALYSE SEMANTIQUE LATENTE (LSA) #############################
###########################################################################################
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Création du modèle de Latent Semantic Analysis 
def prepare_corpus(doc_clean):

    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary,doc_term_matrix

def create_gensim_lsa_model(doc_clean,number_of_topics,words):
 
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    return lsamodel

from collections import Counter

# Tokenisation, stemmisation et suppression des stop words des textes reddit et arxiv
p_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
tokens_r = tokenizer.tokenize(corp_reddit)

en_stop = set(stopwords.words('english'))
tokens_r2 = []
for i in tokens_r :
    tokens_r2.append(i.lower())
    
stopped_tokens_r = [i for i in tokens_r2 if not i in en_stop]
stemmed_tokens_r = [p_stemmer.stem(i) for i in stopped_tokens_r]

tokenizer = RegexpTokenizer(r'\w+')
tokens_a = tokenizer.tokenize(corp_arx)

tokens_a2 = []
for i in tokens_a :
    tokens_a2.append(i.lower())


en_stop = set(stopwords.words('english'))
stopped_tokens_a = [i.lower() for i in tokens_a2 if not i in en_stop]
stemmed_tokens_a = [p_stemmer.stem(i) for i in stopped_tokens_a]

# Application du modele lsa a nos textes afin de comparer les sujets
sujets_reddit = "En ce qui concerne reddit, les 10 principaux sujets sont " +  create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[0][0] + "," + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[1][0] + "," + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[2][0] + "," + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[3][0] + "," + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[4][0]+ "," + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[5][0]+ "," + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[6][0]+ "," + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[7][0]+ ", " + create_gensim_lsa_model([stemmed_tokens_r], 1, 8).show_topic(0)[8][0] +" et "+create_gensim_lsa_model([stemmed_tokens_r], 1, 9).show_topic(0)[9][0]
sujets_arxiv = "En ce qui concerne reddit, les 10 principaux sujets sont " + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[0][0] + "," + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[1][0] + "," + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[2][0] + "," + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[3][0] + "," + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[4][0]+ "," + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[5][0]+ "," + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[6][0]+ "," + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[7][0]+ ", " + create_gensim_lsa_model([stemmed_tokens_a], 1, 8).show_topic(0)[8][0] +" et "   +  create_gensim_lsa_model([stemmed_tokens_r], 1, 9).show_topic(0)[9][0]                                 

# Compteurs des mots les plus utilisés des corpus
comp_red = Counter(stopped_tokens_r).most_common(10)
comp_arx = Counter(stopped_tokens_a).most_common(10)



##########################################################################################
################################### INTERFACE TKINTER ####################################
##########################################################################################
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

# Classe d'analyse avec la méthode TF - IDF
def TFIDF():
    top = tk.Tk()         
    top.title('ANALYSE COMPARATIVE VIA LA METHODE TF-IDF')
    """ """
    figure1 = plt.Figure(figsize=(6,5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, top)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    reddit_scores.plot(kind='bar', legend=True, ax=ax1)
    ax1.set_title('25 Highest TF-IDF Scores for the Reddit Corpus')
    ax1.set_xlabel('Reddit Words')
    ax1.set_ylabel('Word TF-IDF Score')
    
    figure2 = plt.Figure(figsize=(6,5), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, top)
    line2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    arx_scores.plot(kind='bar', legend=True, ax=ax2)
    ax2.set_title('25 Highest TF-IDF Scores for the Arxiv Corpus')
    ax2.set_xlabel('Arxiv Words')
    ax2.set_ylabel('Word TF-IDF Score')
    
    
    
    top.transient(root)       # Réduction popup impossible 
    top.grab_set()          # Interaction avec fenêtre principale impossible

def TFIDF2():
    topd = tk.Tk()          
    topd.title('TABLES ANALYSE COMPARATIVE VIA LA METHODE TF-IDF')
    """ """
    txt221 = ttk.Label(topd, text = "A notre gauche nous avons les scores des 25 meilleurs mots provenant de Reddit et à droite les mots provenant de Arxiv.")
    txt221.pack(side = "top")
    txt222 = ttk.Label(topd, text = reddit_scores)
    txt222.pack(side = "left")
    txt223 = ttk.Label(topd, text = arx_scores)
    txt223.pack(side = "right")


# Classe d'analyse Sémantique Latente
def LSA():
    top2 = tk.Tk()         
    top2.title('GRAPHIQUES ANALYSE COMPARATIVE VIA LA METHODE LSA')
    
    """ """
    txt26 = ttk.Label(top2, text = "L'analyse sémantique latente nous offre déjà quelques informations :")
    txt26.pack(side = "top")
    txt22 = ttk.Label(top2, text = sujets_reddit + ".")
    txt22.pack(side = "top")
    txt25 = ttk.Label(top2, text = sujets_arxiv + ".")
    txt25.pack(side = "top")
    
    dfr = pd.DataFrame(comp_red)
    dfr = dfr.set_index(0)
    figure1 = plt.Figure(figsize=(5,4), dpi=100)
    ax33 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, top2)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    dfr.plot(kind='bar', legend=True, ax=ax33)
    ax33.set_xlabel('Reddit Words')
    ax33.set_ylabel('Word Frequency')
    ax33.set_title('Frequency for the top 10 words (Reddit)')
    
    dfr2 = pd.DataFrame(comp_arx)
    dfr2 = dfr2.set_index(0)
    figure2 = plt.Figure(figsize=(5,4), dpi=100)
    ax3 = figure2.add_subplot(111)
    bar2 = FigureCanvasTkAgg(figure2, top2)
    bar2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    dfr2.plot(kind='bar', legend=True, ax=ax3)
    ax3.set_xlabel('Arxiv Words')
    ax3.set_ylabel('Word Frequency')
    ax3.set_title('Frequency for the top 10 words (Arxiv)')
    
    
    txt266 = ttk.Label(top2, text = "   ")
    txt266.pack(side = "bottom")
    top2.transient(root)       # Réduction popup impossible 
    top2.grab_set()          # Interaction avec fenêtre principale impossible

# Classe d'analyse de Similarité Cosinus
def COS():
    topc = tk.Tk()         
    topc.title('ANALYSE COMPARATIVE VIA LA METHODE SIMILARITE COSINUS')
    """ """
    txt2 = ttk.Label(topc, text = "Nous avons d'abord réaliser une analyse par similarité Cosinus. Voici la matrice résultante :")
    txt2.pack()
    txtm = ttk.Label(topc, text = aa)
    txtm.pack()
    topc.transient(root)       # Réduction popup impossible 
    topc.grab_set()          # Interaction avec fenêtre principale impossible

# Instanciation de la fenêtre principale
root= tk.Tk() 

root.geometry('600x300')
root.title('Résultats Analyse Comparative de Corpus (Reddit et Arxiv)')
txt = ttk.Label(root, text = "Les deux corpus ont été traités et analysés. Voici les différents résultats de l'analyse :")
txt.pack(side = "top")
ttk.Button(root, text='Graphiques Analyse Sémantique Latente', command=LSA).pack(padx=10, pady=10)
ttk.Button(root, text='Analyse Comparative par Similarité Cosinus', command= COS).pack(padx=10, pady=10)
ttk.Button(root, text='Tables Analyse Comparative TF-IDF', command=TFIDF2).pack(padx=10, pady=10)
ttk.Button(root, text='Graphiques Analyse Comparative TF-IDF', command=TFIDF).pack(padx=10, pady=10)

root.mainloop()












