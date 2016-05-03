import numpy as np
import cPickle as pickle

def Preprocess(x,MeanSubtraction=True,ZCA=True):
    print "Pre-processing ..."
    Params = {}
    if MeanSubtraction or ZCA:
        print "Mean subtraction"
        mean = np.mean(x,axis=0)
        std  = np.std(x,axis=0)
        x   -= mean
        x   /= std
        Params["mean"] = mean
        Params["std" ] = std
    if ZCA:
        print "ZCA"
        eps = 1e-5
        C = np.dot(x.T, x) / len(x)
        U, lam, V = np.linalg.svd(C)
        sqlam = np.sqrt(lam+eps)
        Uzca = np.dot( U / sqlam[np.newaxis, :], U.T )
        x = np.dot( x, Uzca.T )
        Params["eps"]  = eps
        Params["Uzca"] = Uzca
        Params["C"] = C
        Params["U"] = U
        Params["V"] = V
        Params["lam"] = lam
    return x, Params

with open("train.pickle","rb") as f:
    d = pickle.load(f)

x,params = Preprocess(d["data"].astype(np.float32)/255.,MeanSubtraction=True,ZCA=True)

d["data_ZCA_byTrainData"]   = x
d["Params_ZCA_byTrainData"] = params
with open("trainZCA.pickle","wb") as f:
    pickle.dump(d,f)
