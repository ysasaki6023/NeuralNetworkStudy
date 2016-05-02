import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm


fig = plt.figure()
ax = {}
ax["DropOut"]    = fig.add_subplot(121)
ax["NoDropOut"]  = fig.add_subplot(122)

dList = {}
dList["DropOut"]   = ["DropOut1","DropOut2","DropOut3"]
dList["NoDropOut"] = ["NoDropOut1","NoDropOut2"]

def myPlot(ax,dName):
    cList = ["black","blue","red","green","cyan"]
    for i,dfile in enumerate(dName):
        print dfile
        d = pd.read_csv("Output_DropTest/%s/output.dat"%dfile)
        dTrain = d[d["mode"]=="Train"]
        dTest  = d[d["mode"]=="Test" ]
        ax.plot(dTrain.epoch, dTrain.accuracy*100., lineStyle="-" , color=cList[i], label=dfile)
        ax.plot(dTest .epoch, dTest .accuracy*100., lineStyle="--", color=cList[i], label="")
    ax.set_xlim(0,50)
    ax.set_ylim(0,100)
    ax.legend(loc=4,fontsize=8)
    ax.grid()
for k in dList:
    myPlot(ax[k],dList[k])
plt.show()
