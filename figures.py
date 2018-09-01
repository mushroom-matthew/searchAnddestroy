#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 00:37:46 2018

@author: getzinmw
"""

import os
import numpy as np
import minecart
import matplotlib.pyplot as plt
from PIL import Image
import PyPDF2

files = os.listdir("../")

n_files = np.size(files)
n = 15
while n < 16:
    """n_files:"""
    """if file[n][-3:-1]"""
    if files[n][-4:] == ".pdf":
        print("GatheringImages:\t\t"+files[n])
        input1 = PyPDF2.PdfFileReader(open("../"+files[n],"rb"))
        page0 = input1.getPage(0)
        xObject = page0['/Resources']['/XObject'].getObject()
    
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()
                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = "RGB"
                else:
                    mode = "P"
    
                if xObject[obj]['/Filter'] == '/FlateDecode':
                    img = Image.frombytes(mode, size, data)
                    img.save(obj[1:] + ".png")
                elif xObject[obj]['/Filter'] == '/DCTDecode':
                    img = open(obj[1:] + ".jpg", "wb")
                    img.write(data)
                    img.close()
                elif xObject[obj]['/Filter'] == '/JPXDecode':
                    img = open(obj[1:] + ".jp2", "wb")
                    img.write(data)
                    img.close()
        
#        doc = minecart.Document(pdffile)
#        page = doc.get_page(0)
#        for image in page.images:
#            im = image.as_pil()
#            im.show()
        
        
        n+=1
        