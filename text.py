# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 22:56:35 2018

@author: matth
"""

import os
import numpy as np
import textract
#from difflib import get_close_matches, SequenceMatcher
#from collections import Counter
#import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim.summarization.summarizer as summ
import json

#def count_close_matches(word,list_text,cutoff):
#    out = get_close_matches(word,list_text,1000,cutoff)
#    return len(out)

files = os.listdir("../")

n_files = np.size(files)
n = 337
while n < n_files:
    """n_files:"""
    """if file[n][-3:-1]"""
    if files[n][-4:] == ".pdf":
        print("\nReadingFile:  "+files[n]+" ("+str(n+1)+" of "+str(n_files)+")")
        text = textract.process("../"+files[n],\
                                method='tesseract',\
                                language='eng',\
                                )
        d_text = text.decode("UTF-8","replace")
#        l_text_1 = re.findall(r"[\w']+",d_text)
#        l_text_2 = d_text.replace("\n"," ")
#        l_text_2.replace("."," ")
#        l_text_2.replace("?"," ")
#        l_text_2 = d_text.split(" ")
#        ll_text_1 = [text.lower() for text in l_text_1]
#        ll_text_2 = [text.lower() for text in l_text_2]
   
        wordcloud_1 = WordCloud(collocations=False,regexp=r"\w[\w'-]+|[0-9]+\s[\w]+").generate(d_text)
        file = open("./"+files[n][0:-4]+"_keywords.txt",'w')
        file.write(json.dumps(wordcloud_1.words_).replace(" ","\n"))
        file.close()
        
#        wordcloud_2 = WordCloud.generate(ll_text_2)
        
        fig1 = plt.figure()
        plt.imshow(wordcloud_1,interpolation='bilinear')
        plt.axis("off")
        plt.show()
        fig1.savefig("./"+files[n][0:-4])
       
        
        ss = summ.summarize(d_text,ratio=0.2)
        file = open("./"+files[n][0 :-3]+"txt",'w')
        file.write(ss)
        file.close()
#        plt.figure()
#        plt.imshow(wordcloud_2,interpolation='bilinear')
#        plt.axis("off")
#        plt.show()
#        print(type(d_text))
#        keywords = ["nanomaterials","biomedical","spectrum","x-ray","gamma",\
#                    "radiation","semiconductor","quantum","reconstruction",\
#                    "geometry","spectral","optogenetics","erg","ecog","photon",\
#                    "patch-clamp","electrophysiology","electroretinography",\
#                    "imaging","therapy","diagnostic","theranostic","protein",\
#                    "delivery","nanoparticle","upconversion","fluorescence",\
#                    "light","visible","k-edge","absorption","antioxidant",\
#                    "oxidative","stress","g-protein","gpcr","opsin",\
#                    "rhodopsin","genetics","energy","scatter","pulse","dose",\
#                    "rf","infrared","nir","electronics","pulse-train",\
#                    "waveform","electricity","electron","neural","network",\
#                    "lightning","radon","particle","wave","microscopy","field",\
#                    "mutation","single-strand","double-strand","free","radical",\
#                    "magnetic","mri","dti","fmri","detector","ccd","emccd",\
#                    "pmt","uv","ultraviolet","dna","eye","retina","genomics",\
#                    "proteomics","scatter","water","fungi","tissue","single-cell",\
#                    "cell","review","abstract","methods","results","conclusions",\
#                    "discussion","cancer","statistics","machine-learning",\
#                    "aperture","grating","interferometry","response","a-wave",\
#                    "b-wave","frequency","damage"]
#        l_m = 0
#        for word in keywords:
#            if len(word) > l_m:
#                l_m = len(word)
#                
#            
#        offset = l_m + 4
#            
#        for word in keywords:
#            num = count_close_matches(word,ll_text_2,0.85)
#            print("Instances of "+word+":"+" "*(offset-len(word))+str(num))
#            
#        counts = Counter(l_text_l)
#        print(counts)
        
    n += 1
    
