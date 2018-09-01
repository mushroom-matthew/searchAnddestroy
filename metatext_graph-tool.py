#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:28:51 2018

@author: getzinmw
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:09:56 2018

@author: matth
"""

import os
import numpy as np
import enchant
#import pandas as pd
from graph_tool.all import * 
#import networkx as nx
#from math import*
#import re
import matplotlib
#import json

def jaccard_similarity(x,y,valx,valy):    
    # This function determines the WEIGHTED similarity between the two papers being compared
    xD = dict(zip(x,valx))
    yD = dict(zip(y,valy))
    intersection_list = set.intersection(*[set(x[0:50]), set(y[0:50])])
    intersection_cardinality = len(intersection_list)*\
    (np.mean([(xD[list(intersection_list)[i]]+yD[list(intersection_list)[i]])\
             for i in range(len(intersection_list))]))
    union_list = set.union(*[set(x[0:50]), set(y[0:50])])
    union_cardinality = len(union_list)
    return intersection_cardinality/float(union_cardinality)

D = enchant.Dict("en_US")

files = os.listdir("../")
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
    
    data = data.lower()
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
        
        val_mat[i,n] = float(v[v_i[i]])
        word_r.append(word_i[v_i[i]])
          
    word_mat.append(word_r)
    
    n+=1
    
    
G = Graph(directed=False)
n = 0
eprop_double = G.new_edge_property("double")

while n < round(n_files):
    for m in range(n+1,round(n_files)):
        sim_mat[n,m] = abs(jaccard_similarity(\
               [word for word in word_mat[n] if D.check(word)],\
               [word for word in word_mat[m] if D.check(word)],\
               [val_mat[i,n] for i in range(np.size(val_mat,0)) if D.check(word_mat[n][i])],\
               [val_mat[i,m] for i in range(np.size(val_mat,0)) if D.check(word_mat[m][i])]))
        if sim_mat[n,m] >= 0.08:
            e = G.add_edge(n,m)
            eprop_double[e] = sim_mat[n,m]
        
    n+=1
G.edge_properties["weight"] = eprop_double
pos = sfdp_layout(G,eweight=G.edge_properties["weight"])    
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="sfdp-sbm-paper-fit.svg")

state_sfdp = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=False)
#state_sfdp_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=True)
state_sfdp.draw(pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[])#, output="sfdp-snbm-paper-fit.svg")
b = state_sfdp.get_bs()[0]

grouping_sfdp = dict(zip(files,b))
print("SFDP Entropy:" + str(state_sfdp.entropy()))




pos = fruchterman_reingold_layout(G,weight=eprop_double)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="fr-sbm-paper-fit.svg")

state_fr = graph_tool.inference.minimize_nested_blockmodel_dl(G, deg_corr=False)
#state_fr_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G, deg_corr=True)
state_fr.draw(pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[])#, output="fr-snbm-paper-fit.svg")
b = state_fr.get_bs()[0]

grouping_fr = dict(zip(files,b))
print("FR Entropy:" + str(state_fr.entropy()))



pos = arf_layout(G,weight=eprop_double)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="arf-sbm-paper-fit.svg")

state_arf = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=False)
#state_arf_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=True)
state_arf.draw(pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[])#, output="arf-snbm-paper-fit.svg")
b = state_arf.get_bs()[0]

grouping_arf = dict(zip(files,b))
print("ARF Entropy:" + str(state_arf.entropy()))




pos = random_layout(G)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="rand-sbm-paper-fit.svg")

state_rand = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=False)
#state_rand_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=True)
state_rand.draw(pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[])#, output="rand-snbm-paper-fit.svg")
b = state_rand.get_bs()[0]

grouping_rand = dict(zip(files,b))
print("Random Entropy:" + str(state_rand.entropy()))
#pos = radial_tree_layout(G)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="radtree-sbm-paper-fit.svg")
#
#state = graph_tool.inference.minimize_nested_blockmodel_dl(G)
#state.draw(pos=pos, output="radtree-snbm-paper-fit.svg")

#pos = planar_layout(G)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="planar-sbm-paper-fit.svg")
#
#state = graph_tool.inference.minimize_nested_blockmodel_dl(G)
#state.draw(pos=pos, output="planar-snbm-paper-fit.svg")
#elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.25*5]
#esmall=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.25*5 and d['weight'] >0.2*5]]
#esmaller=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.2*5 and d['weight'] >0.15*5]]
#esmallerer=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.15*5 and d['weight'] >0.10*5]]
#esmallest=[(u,v) for (u,v,d) in G.edges(data=True) if [d['weight'] <=0.1*5]]
#
#pos=nx.spring_layout(G,dim=2,k=None,scale=100) # positions for all nodes
#
## nodes
#nx.draw_networkx_nodes(G,pos,node_size=50)
#
## edges
#nx.draw_networkx_edges(G,pos,edgelist=esmallest,
#                    width=0.25,alpha=0.1,edge_color='k',style='dashed')
#nx.draw_networkx_edges(G,pos,edgelist=esmallerer,
#                    width=0.25,alpha=0.1,edge_color='g',style='dashed')
#nx.draw_networkx_edges(G,pos,edgelist=esmaller,
#                    width=0.25,alpha=0.1,edge_color='r',style='dashed')
#nx.draw_networkx_edges(G,pos,edgelist=esmall,
#                    width=0.25,alpha=0.1,edge_color='b',style='dashed')
#nx.draw_networkx_edges(G,pos,edgelist=elarge,
#                    width=1,color='o')
#
## labels
#nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')
##plt.xlim((-10,10))
##plt.ylim((-10,10))
