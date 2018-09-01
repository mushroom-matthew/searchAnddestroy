#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:06:24 2018

@author: getzinmw
"""

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

D = enchant.Dict("en_US")
needed_words = ("optogenetic","photostimulation","nanomaterials","biomedical","spectrum","x-ray","gamma",\
      "radiation","semiconductor","quantum","reconstruction",\
      "geometry","spectral","optogenetics","erg","ecog","photon",\
      "patch-clamp","electrophysiology","electroretinography",\
      "imaging","therapy","diagnostic","theranostic","protein",\
      "delivery","nanoparticle","upconversion","fluorescence",\
      "light","visible","k-edge","absorption","antioxidant",\
      "oxidative","stress","g-protein","gpcr","opsin","halorhodopsin","archaerhodopsin",\
      "rhodopsin","genetics","energy","scatter","pulse","dose",\
      "rf","infrared","nir","electronics","pulse-train",\
      "waveform","electricity","electron","neural","network",\
      "lightning","radon","particle","wave","microscopy","field",\
      "mutation","single-strand","double-strand","free","radical",\
      "magnetic","mri","dti","fmri","detector","ccd","emccd",\
      "pmt","uv","ultraviolet","dna","eye","retina","genomics",\
      "proteomics","scatter","water","fungi","tissue","single-cell",\
      "cell","review","abstract","methods","results","conclusions",\
      "discussion","cancer","statistics","machine-learning",\
      "aperture","grating","interferometry","response","a-wave",\
      "b-wave","frequency","damage")
for word in needed_words:
    if not D.check(word):
        D.add(word)
        print(word+" is "+str(D.check(word)))
        

#%%
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
  
#print(word_mat)
G = Graph(directed=False)
n = 0
eprop_double = G.new_edge_property("double")
vprop_string = G.new_vertex_property("string")
v = list()
while n < round(n_files):
    if n == 0:
        v.append(G.add_vertex())
        vprop_string[v[n]] = files[n][0:15]
    for m in range(n+1,round(n_files)):
        if n ==0:
            v.append(G.add_vertex())
            vprop_string[v[m]] = files[m][0:15]
        sim_mat[n,m] = abs(jaccard_similarity(\
               [word for word in word_mat[n] if D.check(word)],\
               [word for word in word_mat[m] if D.check(word)],\
               [val_mat[i,n] for i in range(len(word_mat[n])) if D.check(word_mat[n][i])],\
               [val_mat[i,m] for i in range(len(word_mat[m])) if D.check(word_mat[m][i])]))
        if sim_mat[n,m] >= 0.5:
            e = G.add_edge(v[n],v[m])
            eprop_double[e] = sim_mat[n,m]
        
    n+=1
    
G.edge_properties["weight"] = eprop_double
G.vertex_properties["names"] = vprop_string
pos = sfdp_layout(G,eweight=G.edge_properties["weight"])    
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="sfdp-sbm-paper-fit.svg")
#%%
state_sfdp = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=False)
#state_sfdp_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=True)
draw_hierarchy(state_sfdp,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[],vertex_text=G.vertex_properties["names"],vertex_text_position="centered",vertex_font_size=7,output_size=(1200,1200), output="sfdp-snbm-paper-fit.svg")
b = state_sfdp.get_bs()[0]

grouping_sfdp = dict(zip(files,b))
print("SFDP Entropy:" + str(state_sfdp.entropy()))




pos = fruchterman_reingold_layout(G,weight=eprop_double)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="fr-sbm-paper-fit.svg")

state_fr = graph_tool.inference.minimize_nested_blockmodel_dl(G, deg_corr=False)
#state_fr_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G, deg_corr=True)
draw_hierarchy(state_fr,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[],vertex_text=G.vertex_properties["names"],vertex_text_position="centered",vertex_font_size=7,output_size=(1200,1200), output="fr-snbm-paper-fit.svg")
b = state_fr.get_bs()[0]

grouping_fr = dict(zip(files,b))
print("FR Entropy:" + str(state_fr.entropy()))



pos = arf_layout(G,weight=eprop_double)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="arf-sbm-paper-fit.svg")

state_arf = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=False)
#state_arf_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=True)
draw_hierarchy(state_arf,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[],vertex_text=G.vertex_properties["names"],vertex_text_position="centered",vertex_font_size=7,output_size=(1200,1200), output="arf-snbm-paper-fit.svg")
b = state_arf.get_bs()[0]

grouping_arf = dict(zip(files,b))
print("ARF Entropy:" + str(state_arf.entropy()))




pos = random_layout(G)
#state = graph_tool.inference.minimize_blockmodel_dl(G)
#state.draw(pos=pos, output="rand-sbm-paper-fit.svg")

state_rand = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=False)
#state_rand_dc = graph_tool.inference.minimize_nested_blockmodel_dl(G,deg_corr=True)
draw_hierarchy(state_rand,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           edge_gradient=[],vertex_text=G.vertex_properties["names"],vertex_text_position="centered",vertex_font_size=7,output_size=(1200,1200), output="rand-snbm-paper-fit.svg")
b = state_rand.get_bs()[0]

grouping_rand = dict(zip(files,b))
print("Random Entropy:" + str(state_rand.entropy()))