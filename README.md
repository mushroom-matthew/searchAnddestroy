# searchAnddestroy
Python-based word parser and graph-tool for pdfs

figures.py --- half attempt at finding and compiling the information in the figures of pdfs
               implements packages : os, numpy, minecart, matplotlib.pyplot, PIL, PyPDF2
               


graph_analysis.py --- has examples of loading saved graphs from metatext analysis for further analysis based on grouping
                      implements packages : os, numpy, enchant, grap_tool.all, math, matplotlib(.pyplot), json, wordcloud, itertools
                      
                                          
metatext_graph-tool.py --- builds undirected graph(s) from .txt files parsed from pdfs with text.py
                           the strength of the connection between nodes is determined by a modified Jaccard Similarity metric
                           of sets of words between papers (future iterations should look to use time in directed graphs)
                           implements packages: os, numpy, enchant, graph_tool.all, matplotlib
                           
         
metatext_graph-tool_nltk.py --- explores .txt files pared from pdfs with text.py using Natural Language Toolkit (half attempt)
                                implements packages : os, numpy, enchant, grap_tool.all, math, matplotlib(.pyplot), json, wordcloud, nltk
                                
                                

metatext_grouping.py --- builds undirected graph from .txt files parsed from pdf with text.py using networkx 
                         implements packages: os, numpy, networkx
                         
                         
metatext_summaries2wordcloud2gt.py --- builds undirected graph(s) from .txt files parsed from pdfs with text.py
                           the strength of the connection between nodes is determined by a modified Jaccard Similarity metric
                           there is a modified dictionary for specific words that should be added to the dictionary if not already
                              considered a word
                           implements packagages: os, numpy, enchant, graph_tool.all, math, matplotlib, json, wordcloud
                           
                           
                           
text.py --- harvest words from pdfs using textract, makes wordcloud and keyword.txt file with wordcloud, generates summary.txt file with
            gensim summarization tools
            implements packages: os, numpy, textract, wordcloud, matplotlib.pyplot, gensim.summarization.summarizer, json
