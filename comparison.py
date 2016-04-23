import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read(fName,color,label,rolling=None):
    d = pd.read_csv(fName)
    dTest  = d[d.Mode=="test" ]
    dTrain = d[d.Mode=="train"]

    xTrain = dTrain.fAccuracy * 100.
    xTest  = dTest .fAccuracy * 100.

    if rolling:
        xTest  = xTest .rolling(window=rolling,center=False).mean()
        xTrain = xTrain.rolling(window=rolling,center=False).mean()
    
    plt.plot(dTrain.Epoch, xTrain , color=color, linestyle="-" , label=label)
    plt.plot(dTest.Epoch , xTest  , color=color, linestyle="--")
    #plt.show()
    
fList = ["0.0","0.2","0.4","0.6","0.8"]

plt.axis([0,200,0.1,100])
for i,f in enumerate(fList):
    read("rec/N1=100_N2=100_Nf=16_DR=%s_v2.dat"%f, color=cm.gist_ncar(float(i)/len(fList)), label="DropOut=%s"%f, rolling=5)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.yscale("log")
plt.show()
