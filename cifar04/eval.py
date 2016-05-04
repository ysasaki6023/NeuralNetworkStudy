#!/usr/bin/env python
import argparse
import time

import numpy as np
import six
import os
import shutil

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from chainer.utils import conv
from chainer.serializers.hdf5 import HDF5Serializer, HDF5Deserializer

import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm
import h5py
import ast
import imp

def CifarEval(fileName=None,batchsize = 1000):
    id_gpu  = 0

    OutStr = ""
    OutStr += 'GPU: {}\n'.format(id_gpu)
    OutStr += 'Minibatch-size: {}\n'.format(batchsize) 
    OutStr += '' 

    print OutStr

    fFolder, _ = os.path.split(fileName)
    fInput     = fileName
    fOutput = open(os.path.join(fFolder,"classify.html"),"w")

# Prepare dataset
    data_ev  = np.zeros((10000,3*32*32),dtype=np.float32)
    label_ev = np.zeros((10000),dtype=np.int32)
    nameList = []#np.array((10000),dtype=np.str)
    I_colors = 3
    I_Xunit  = 32
    I_Yunit  = 32
    F_unit   = 100   # be careful!!

    h5f_ev = h5py.File("data_cifar100/test.h5f","r")
    data_ev[:]  = h5f_ev["ZCA_byTrainData/data"].value
    label_ev[:] = h5f_ev["Info/fine_labels"].value

    with open("data_cifar100/batches.meta","rb") as f:
        tmp = pickle.load(f)
        nameList  = tmp["fine_label_names"]

## Prep
    x_ev = data_ev.reshape((len(data_ev),3,32,32))
    y_ev = label_ev
    N_ev = len(data_ev) # 10000

    print 'Load optimizer state from %s'%(fileName)
    hFile = h5py.File(fileName,"r")
    sFile = HDF5Deserializer(hFile)
    kwd = ast.literal_eval((hFile["kwd"].value))

    myNet = hFile["net"].value
    net = imp.new_module('net')
    exec myNet in net.__dict__

    model,ModelKwd = net.GenModel(**kwd)
    if id_gpu >= 0:
        cuda.get_device(id_gpu).use()
        model.to_gpu()
    xp = np if id_gpu < 0 else cuda.cupy

# Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

# Init/Resume
    sFile["model"].load(model)

# Learning loop
    result = np.zeros( (N_ev, F_unit), dtype=np.float32)
    tStr = ""
    tStr += "<table border=1>"
    # evaluation
    for i in six.moves.range(0, N_ev, batchsize):
        x = chainer.Variable(xp.asarray(x_ev[i:i + batchsize]),volatile='on')
        t = chainer.cuda.to_cpu(chainer.Variable(xp.asarray(y_ev[i:i + batchsize]),volatile='on').data)
        y = chainer.cuda.to_cpu(F.softmax((model.predictor(x))).data)
        for k,j in enumerate(y):
            tStr += "<tr>"
            tStr += "<td>%d</td>"%(k+i)
            tStr += '<td><img src="%s" width=96px height=96px> </img></td><td>'%(os.path.join(os.getcwd(),"data_cifar100/img/test_%d.jpg"%(k+i)))
            s = np.argsort(j)
            for p in s[::-1][:5]:
                if p == t[k]:
                    tStr += "<b> %s (%.1f%%) </b> <br>"%(nameList[p], j[p]*100.)
                else:
                    tStr += "%s (%.1f%%) <br>"%(nameList[p], j[p]*100.)
            tStr += "</td><td>"
            tStr += "<b>%s</b>"%nameList[t[k]]
            tStr += "</td></tr>"
            #print j
            #tStr += str(list(j))
    tStr += "</table>"

    hstr = """
<!DOCTYPE html>
<html>
<head><title>Image classification</title></head>
<body>
<h1>%s</h1>
%s
</body></html>
"""%(fInput,tStr)
    fOutput.write(hstr)
    if fOutput: fOutput.close()

if __name__=="__main__":
    CifarEval(fileName="Output/L4/mlp_22.hdf5")
