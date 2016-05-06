import os,sys,shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

f = open("AOoutput.dat")
myDict = None
for line in f.readlines():
    #print line
    line = line.split(",")
    val = line[0]
    dic = ast.literal_eval(",".join(line[1:]).replace(" ",""))
    print val
    if myDict==None:
        myDict = {x:[float(dic[x])] for x in dic}
        myDict["accuracy"] = [float(val)]
    else:
        myDict["accuracy"].append(float(val))
        for i in dic:
            myDict[i].append(float(dic[i]))
d = pd.DataFrame.from_dict(myDict)
print d
#print np.array(d.accuracy),np.array(d.P1C_filter)
plt.scatter(d.P0P_ksize,d.accuracy)
#plt.scatter(d.P0C_feature,d.accuracy)
plt.show()
