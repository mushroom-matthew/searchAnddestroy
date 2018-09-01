#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:02:08 2018

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
import matplotlib.pyplot as plt
import itertools


G = load_graph("graph_paper_connection_custom_library.xml.gz")
y = G.edge_properties["weight"].copy()
z = G.edge_properties["weight"].copy()
y.a = np.log(y.a)
pos = arf_layout(G,weight=G.edge_properties["weight"])
state = minimize_nested_blockmodel_dl(G, state_args=dict(recs=[y],
                                                               rec_types=["real-normal"]))

names = G.vertex_properties["names"]
print("Number of vertices: " +str(G.num_vertices()))
vsum = graph_tool.incident_edges_op(G, "out", "sum", G.edge_index)
print([names[ii] for ii in range(len(vsum.a)) if vsum.a[ii] == 0])
n=0
while n < G.num_vertices():
    vsum = graph_tool.incident_edges_op(G, "out", "sum", G.edge_index)
    if vsum.a[n] == 0:
        G.remove_vertex(n)
        continue
    
    n+=1
print("Number of vertices: " +str(G.num_vertices()))


#%%
bs = state.get_bs()
bs += [np.zeros(1)]*(10-len(bs))

state = state.copy(bs = bs, sampling=True)

dS, nattempts, nmoves = state.mcmc_sweep(niter=1000)

print("Change in description length:", dS)
print("Number of accepted vertex moves:", nmoves)
#%% 
mcmc_equilibrate(state,wait=1000,mcmc_args=dict(niter=10))

pv = [None] * len(state.get_levels())

def collect_marginals(s):
   global pv
   pv = [sl.collect_vertex_marginals(pv[l]) for l, sl in enumerate(s.get_levels())]

state_level_marginals = mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                    callback=collect_marginals)

draw_hierarchy(state,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           vertex_shape="pie", vertex_pie_fractions=pv[0],edge_gradient=[])

#%% 
h = [np.zeros(G.num_vertices() + 1) for s in state.get_levels()]

def collect_num_groups(s):
    for l, sl in enumerate(s.get_levels()):
       B = sl.get_nonempty_B()
       h[l][B] += 1

# Now we collect the marginal distribution for exactly 100,000 sweeps
state_group_marginals = mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                    callback=collect_num_groups)
draw_hierarchy(state,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           vertex_shape="pie",vertex_pie_fractions=pv[0],vertex_text=G.vertex_properties["names"],vertex_text_position="centered",
           vertex_font_size=7.5,edge_gradient=[],output="arf_mcmc_eq_sweep_10000_final.svg")


for ii in range(np.size(h,0)):
    plt.subplot(2,5,ii+1)
    l0_non_zero = h[ii][np.min(np.nonzero(h[ii])):np.max(np.nonzero(h[ii]))+1]
    plt.bar(list(itertools.chain.from_iterable(np.nonzero(h[ii]))),l0_non_zero/np.sum(l0_non_zero),)
    plt.xticks(list(itertools.chain.from_iterable(np.nonzero(h[ii]))))


#%%
for i in range(1):
    state.mcmc_sweep(niter=1000)
    draw_hierarchy(state,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
           vertex_shape="pie",edge_gradient=[])

#%%
#y = G.edge_properties["weight"].copy()
#z = G.edge_properties["weight"].copy()
#y.a = np.log(y.a)
#
#state_exp = minimize_nested_blockmodel_dl(G, state_args=dict(recs=[z],
#                                                               rec_types=["real-exponential"]))
#draw_hierarchy(state_exp,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
#           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
#           vertex_shape="pie",vertex_pie_fractions=pv[0],edge_gradient=[],vertex_size=14,vertex_text=G.vertex_properties["names"],vertex_text_position="centered",vertex_font_size=7,output_size=(1200,1200),output="rand_snbm_paper_real_exponential.svg")
#
#state_ln = minimize_nested_blockmodel_dl(G, state_args=dict(recs=[y],
#                                                               rec_types=["real-normal"]))
#draw_hierarchy(state_ln,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
#           eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
#           vertex_shape="pie",vertex_pie_fractions=pv[0],edge_gradient=[],vertex_size=14,vertex_text=G.vertex_properties["names"],vertex_text_position="centered",vertex_font_size=7,output_size=(1200,1200),output="rand_snbm_paper_real_normal.svg")
#
#L1 = -state_exp.entropy()
#L2 = -state_ln.entropy() - np.log(y.a[np.sign(y.a)>0]).sum()
#
#print(u"ln \u039b: ", L2 - L1) #log-normal is better fit
#%%
bs = state.get_bs()
g0_allpaps = list()
for ii in range(np.max(np.unique(bs[0]))+1):
    indices = np.where(bs[0]==ii)
    g0_paps = list()
    for index in indices[0]:
        g0_paps.append(G.vertex_properties["names"][index])

    g0_allpaps.append(g0_paps) 
    
#%%

for v in G.vertices():
    
#    for e in v.all_edges():
#        print(e)
    for w in v.all_neighbors():
#        print(w)
        print(G.vertex_properties["names"][v] + " in group " +str(bs[0][G.vertex_index[v]]) + " is connected to " + G.vertex_properties["names"][w] + " in group "+str(bs[0][G.vertex_index[w]])) 
       
#   for e, w in zip(v.out_edges(), v.out_neighbors()):
#       assert e.target() == w
        
#%%
files = os.listdir("./summaries/")
n=0;
for paplist in g0_allpaps:
#    print(paplist)
    master_metatext = ''
    for pap in paplist:
        
        filesp = [file for file in files if file[0:15] == pap]
        print(filesp)
        
        text = open("./summaries/"+filesp[0],'r')
            
        master_metatext += text.read()
        text.close()
        
    wordcloud_1 = WordCloud(collocations=False,regexp=r"\w[\w'-]+|[0-9]+\s[\w]+").generate(master_metatext)
    fig1 = plt.figure()
    plt.imshow(wordcloud_1,interpolation='bilinear')
    plt.axis("off")
    plt.show()
    fig1.savefig("./arf-group-"+str(n)+"-wordmaps.png")
    n+=1
    
#%%
import scipy
B = modularity_matrix(G)
A = B*np.identity(B.shape[0])
ew,ev = scipy.linalg.eig(A)
plt.figure(figsize=(8, 2))
plt.scatter(np.real(ew), np.imag(ew), c=np.sqrt(np.abs(ew)), linewidths=0, alpha=0.6)
plt.xlabel(r"$\operatorname{Re}(\lambda)$")
plt.ylabel(r"$\operatorname{Im}(\lambda)$")
plt.tight_layout()
#%%
pr = pagerank(G)
draw_hierarchy(state,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.afmhot, .6),
          eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
          vertex_fill_color=pr,vertex_size=prop_to_size(pr,mi=5,ma=15),vorder=pr,vcmap=matplotlib.cm.gist_heat,edge_gradient=[])


#%%
vp,ep = betweenness(G)
draw_hierarchy(state,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.afmhot, .6),
          eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(ep, mi=0.5, ma=5),
          vertex_fill_color=vp,vertex_size=prop_to_size(vp,mi=5,ma=15),vorder=vp,vcmap=matplotlib.cm.gist_heat,edge_gradient=[])

#%%
c = closeness(G)
pos = radial_tree_layout(G,G.vertex(c))
graph_draw(G,pos=pos,edge_color=prop_to_size(G.edge_properties["weight"], power=1, log=True), ecmap=(matplotlib.cm.afmhot, .6),
    eorder=G.edge_properties["weight"], edge_pen_width=prop_to_size(G.edge_properties["weight"], 1, 4, power=1, log=True),
    vertex_fill_color=c,vertex_size=prop_to_size(c,mi=5,ma=15),vorder=c,vcmap=matplotlib.cm.gist_heat,edge_gradient=[])

#%%
