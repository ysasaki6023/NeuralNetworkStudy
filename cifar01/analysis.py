import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm


fig = plt.figure()
ax = {}
ax["Normalize"]  = fig.add_subplot(321)
ax["FilterSize"] = fig.add_subplot(322)
ax["Features"]   = fig.add_subplot(323)
ax["Units"]      = fig.add_subplot(324)
ax["DropOut"]    = fig.add_subplot(325)
ax["Ksize"]      = fig.add_subplot(326)

dList = {}
dList["Normalize"]  = ["Baseline", "NormalzeOnOnlyFirstLayer", "NormalzeOnAll"]
dList["FilterSize"] = ["Baseline", "Filter555", "Filter777"]
dList["Features"]   = ["Baseline", "Feature333", "Feature444", "Feature654", "Feature666"]
dList["Units"]      = ["Baseline", "Unit50", "Unit100", "Unit500", "Unit1000"]
dList["DropOut"]    = ["Baseline", "DropOutL0.5L0.5","DropOutL0.2L0.2","DropOutL0.5L0.0","DropOutL0.0L0.5"]
dList["Ksize"]      = ["Baseline", "Ksize333", "Ksize444"]

def myPlot(ax,dName):
    cList = ["black","blue","red","green","cyan"]
    for i,dfile in enumerate(dName):
        print dfile
        d = pd.read_csv("Output/%s/output.dat"%dfile)
        dTrain = d[d["mode"]=="Train"]
        dTest  = d[d["mode"]=="Test" ]
        ax.plot(dTrain.epoch, dTrain.accuracy*100., lineStyle="-" , color=cList[i], label=dfile)
        ax.plot(dTest .epoch, dTest .accuracy*100., lineStyle="--", color=cList[i], label="")
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.legend(loc=4,fontsize=8)
for k in dList:
    myPlot(ax[k],dList[k])
plt.show()
