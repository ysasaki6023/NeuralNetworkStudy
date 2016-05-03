import numpy as np
import cPickle as pickle

def Preprocess(x,Params):
    print "Pre-processing ..."
    MeanSubtraction = True
    ZCA = True
    if MeanSubtraction or ZCA:
        print "Mean subtraction"
        mean = Params["mean"]
        std  = Params["std"]
        x   -= mean
        x   /= std
        Params["mean"] = mean
        Params["std" ] = std
    if ZCA:
        print "ZCA"
        eps = Params["eps"]
        C = Params["C"]
        U   = Params["U"]
        lam = Params["lam"]
        V   = Params["V"]
        sqlam = np.sqrt(lam+eps)
        Uzca= Params["Uzca"]
        x = np.dot( x, Uzca.T )
        Params["eps"]  = eps
        Params["Uzca"] = Uzca
        Params["C"] = C
        Params["U"] = U
        Params["V"] = V
        Params["lam"] = lam
    return x, Params

with open("trainZCA.pickle","rb") as f:
    dParam = pickle.load(f)
with open("testZCA.pickle","rb") as f:
    d = pickle.load(f)

x,params = Preprocess(d["data"].astype(np.float32)/255.,dParam["Params_ZCA_byTrainData"])

d["data_ZCA_byTrainData"]   = x
d["Params_ZCA_byTrainData"] = params
with open("testtrainZCA.pickle","wb") as f:
    pickle.dump(d,f)
