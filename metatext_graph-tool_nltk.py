# -*- coding: utf-8 -*-
import os
import numpy as np
import enchant
#import pandas as pd
from graph_tool.all import * 
#import networkx as nx
from math import*
#import re
import matplotlib
import json
from wordcloud import WordCloud
import nltk

def jaccard_similarity(x,y,valx,valy):    
    # This function determines the WEIGHTED similarity between the two papers being compared
    xD = dict(zip(x,valx))
    yD = dict(zip(y,valy))
    intersection_list = set.intersection(*[set(x[0:50]), set(y[0:50])])
    intersection_cardinality = len(intersection_list)*\
            (np.sum([xD[list(intersection_list)[i]]+yD[list(intersection_list)[i]]\
            for i in range(len(intersection_list))]))    
    union_list = set.union(*[set(x[0:50]), set(y[0:50])])
    union_cardinality = len(union_list)
    return intersection_cardinality/float(union_cardinality)

files = os.listdir("./summaries/")
files = [ fi for fi in files if fi.endswith(".txt")]
n_files = np.size(files)

word_mat = list(list())
val_mat = np.zeros((200,n_files))
sim_mat = np.zeros((n_files,n_files))

n=0
while n < n_files:
    """n_files:"""
    """if file[n][-3:-1]"""
    print("\nReadingFile:  "+files[n]+" ("+str(n+1)+" of "+str(n_files)+")")
    f_open = open("./summaries/"+files[n],'r')
    text = f_open.read()
    f_open.close()
#    d_text = text.decode("UTF-8","replace")
#        l_text_1 = re.findall(r"[\w']+",d_text)
#        l_text_2 = d_text.replace("\n"," ")
#        l_text_2.replace("."," ")
#        l_text_2.replace("?"," ")
#        l_text_2 = d_text.split(" ")
#        ll_text_1 = [text.lower() for text in l_text_1]
#        ll_text_2 = [text.lower() for text in l_text_2]
   
    wordcloud_1 = WordCloud(collocations=False,regexp=r"\w[\w'-]+|[0-9]+\s[\w]+").generate(text)
    
    word_jumble = json.dumps(wordcloud_1.words_)
    
    data = word_jumble.lower()
#    data = data.replace("\n"," ")
    x = data.split("\"")
    
    word_i = x[1::2]
    val_it = x[2::2]
    val_it = [val_t.replace(",","") for val_t in val_it]
    val_it = [val_t.replace(": ","") for val_t in val_it]
    
    v = [float(val_t.replace("}","")) for val_t in val_it]
    
    v_i = sorted(range(len(v)), key=lambda k: v[k], reverse=True)
#    for i in v_i:
#        print(word_i[i]+"--"+str(float(v[i])*1e-16))

    word_r = []
    for i in range(len(v_i)):
        
        val_mat[i,n] = float(v[v_i[i]])
        word_r.append(word_i[v_i[i]])
          
    word_mat.append(word_r)
    
    n+=1