import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Load(fileName):
    print fileName
    d = pd.read_csv(fileName)
    dTest  = d[d["mode"]=="Test"]
    dTrain = d[d["mode"]=="Train"]
    return dTest,dTrain


fNames = ["L3_ksize322","L4","L4_ksize222","L4_unit100","L4_unit1000","L4_unit200","L6_ksize222","L8","L8_Unit100"]
#fNames = ["L3_ksize322","L4","L4_ksize222","L4_unit100","L4_unit200","L6_ksize222","L8","L8_Unit100"]
#fNames = ["L6_ksize222","L8","L8_Unit100"]
fig, axes = plt.subplots(3,3)
for i,row in enumerate(axes):
    for j, cell in enumerate(row):
        index = i*len(row) + j
        if index>=len(fNames): continue
        name  = fNames[index]
        dTest,dTrain = Load("Output/"+name+"/output.dat")
        cell.set_title(name)
        cell.plot(dTrain["epoch"],dTrain["accuracy"]*100,"-" )
        cell.plot(dTest ["epoch"],dTest ["accuracy"]*100,"--")
        cell.set_ylim(0,100)
        cell.grid()

fig.show()
raw_input()
