# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:09:56 2018

@author: matth
"""

import os
import numpy as np
#import pandas as pd
#import 
import networkx as nx
#from math import*
#import re
#import matplotlib.pyplot as plt
#import json

def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

files = os.listdir("./")
files = [ fi for fi in files if fi.endswith("_keywords.txt")]
n_files = np.size(files)

word_mat = list(list())
val_mat = np.zeros((200,n_files))
sim_mat = np.zeros((n_files,n_files))

n = 0
while n < n_files:
#    print(files[n])
    o_fi = open(files[n],'r')
    data = o_fi.read()
    o_fi.close()
    
    data = data.replace("\n"," ")
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
        
        val_mat[i,n] = float(v[v_i[i]]*1e-2)
        word_r.append(  word_i[v_i[i]]   )
          
    word_mat.append(word_r)
    
    n+=1
    
    
G = nx.Graph()
n = 0
while n < round(n_files):
    for m in range(n,round(n_files)):
        sim_mat[n,m] = abs(jaccard_similarity(word_mat[n],word_mat[m]))
        if sim_mat[n,m] >= 0.05:
            G.add_edge(str(n),str(m),weight=sim_mat[n,m]*5)
        
    n+=1
    
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.25*5]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.25*5 and d['weight'] >0.2*5]]
esmaller=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.2*5 and d['weight'] >0.15*5]]
esmallerer=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.15*5 and d['weight'] >0.10*5]]
esmallest=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.1*5]]

pos=nx.spring_layout(G,dim=2,k=None,scale=100) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=50)

# edges
nx.draw_networkx_edges(G,pos,edgelist=esmallest,
                    width=0.25,alpha=0.1,edge_color='k',style='dashed')
nx.draw_networkx_edges(G,pos,edgelist=esmallerer,
                    width=0.25,alpha=0.1,edge_color='g',style='dashed')
nx.draw_networkx_edges(G,pos,edgelist=esmaller,
                    width=0.25,alpha=0.1,edge_color='r',style='dashed')
nx.draw_networkx_edges(G,pos,edgelist=esmall,
                    width=0.25,alpha=0.1,edge_color='b',style='dashed')
nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=1,color='o')

# labels
nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')
#plt.xlim((-10,10))
#plt.ylim((-10,10))
