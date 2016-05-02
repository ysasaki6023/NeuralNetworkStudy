import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from PIL import Image

fName = "test.pickle"

with open(fName,"rb") as f:
    d = pickle.load(f)

imgList = d["data"]
imgList = imgList.reshape((10000,3,32,32)).transpose(0,2,3,1)

for i in range(len(imgList)):
    if i%100==0: print i
    pImg = Image.fromarray(imgList[i])
    pImg.save("img/test_%d.jpg"%i)
