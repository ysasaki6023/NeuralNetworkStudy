import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm

fName = "output.dat"
d = pd.read_csv(fName)

dTrain = d[d["mode"]=="Train"]
dTest  = d[d["mode"]=="Test" ]

plt.ylim(80,100)

plt.plot(dTrain.epoch, dTrain.accuracy*100., lineStyle="-" , label="Train Data")
plt.plot(dTest .epoch, dTest .accuracy*100., lineStyle="--", label="Test  Data")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.legend(loc=4)

plt.show()
