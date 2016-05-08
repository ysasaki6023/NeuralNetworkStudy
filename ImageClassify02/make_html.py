# -*- coding: utf-8 -*-
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
import net

import sys, codecs
#なぜか次の行のやつが通らないことがある。versionの問題かな？
#sys.stdout = codecs.EncodedFile(sys.stdout, 'utf_8')
sys.stdout = codecs.lookup(u'utf_8')[-1](sys.stdout)
print "sys.getdefaultencoding() => ",sys.getdefaultencoding()
print "sys.stdout.encoding => ",sys.stdout.encoding

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
    h5f = h5py.File("input.dat","r")
    data     = h5f["Original/Data"].value
    label    = h5f["Info/Label"].value
    imIndex  = h5f["Info/ImIndex"].value
    agIndex  = h5f["Info/AgIndex"].value
    nameListName  = h5f["Info/imageClassName" ].value
    agIndex  = h5f["Info/AgIndex"].value
    imageURL = h5f["Info/imageURL"].value
    nameList = []
    #nameList = [""]*len(nameListName)
    for i,n in enumerate(nameListName):
        #nameList.append(n)
        #a = "あああ" +n.encode("utf-8")
        #a = "あああ"
        nameList.append(unicode(n))
    print nameList

    print 'Load evaluation dataset from %s'%(os.path.dirname(fileName)+"/TraEva.hdf5")
    with h5py.File(os.path.dirname(fileName)+"/TraEva.hdf5","r") as f:
        TraEva = f["TraEva"].value
    print TraEva
    print type(TraEva)
    #kwd = ast.literal_eval((hFile["kwd"].value))

    data  = data[np.logical_not(TraEva)]
    label = label[np.logical_not(TraEva)]
    agIndex_ev = agIndex[np.logical_not(TraEva)]
    data_ev  = data [agIndex_ev == 0]
    label_ev = label[agIndex_ev == 0]
    imgURL_ev= imageURL[np.logical_not(TraEva[::3])]
    rr = np.random.permutation(len(data_ev))
    #kwd = ast.literal_eval((hFile["kwd"].value))
    data_ev  = data_ev [rr[:2000]]
    label_ev = label_ev[rr[:2000]]
    imgURL_ev= imgURL_ev[rr[:2000]]

    I_colors = 3
    I_Xunit  = 32
    I_Yunit  = 32
    F_unit   = max(label)+1   # be careful!!

    data_ev -= np.mean(data_ev,axis=1).reshape((len(data_ev),1))
    data_ev -= np.mean(data_ev,axis=0)
    data_ev /= np.std (data_ev,axis=0)

## Prep
    x_ev = data_ev.reshape((len(data_ev),3,32,32))
    y_ev = label_ev
    N_ev = len(data_ev) # 10000

    print 'Load optimizer state from %s'%(fileName)
    hFile = h5py.File(fileName,"r")
    sFile = HDF5Deserializer(hFile)
    #kwd = ast.literal_eval((hFile["kwd"].value))
    kwd = {}

    """
    myNet = hFile["net"].value
    net = imp.new_module('net')
    exec myNet in net.__dict__
    """

    model = net.GenModel(F_unit = (len(nameList)))
    id_gpu = -1
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
    rightTot = 0
    right1   = 0
    right5   = 0
    for i in six.moves.range(0, N_ev, batchsize):
        x = chainer.Variable(xp.asarray(x_ev[i:i + batchsize]),volatile='on')
        t = chainer.cuda.to_cpu(chainer.Variable(xp.asarray(y_ev[i:i + batchsize]),volatile='on').data)
        y = chainer.cuda.to_cpu(F.softmax((model.predictor(x))).data)
        print i
        for k,j in enumerate(y):
            rightTot += 1
            tStr += "<tr>"
            tStr += "<td>%d</td>"%(k+i)
            tStr += '<td><img src="%s" width=96px height=96px> </img></td><td>'%(os.path.join(os.getcwd(),"data/%s"%(imgURL_ev[i+k])))
            s = np.argsort(j)
            for n,p in enumerate(s[::-1][:5]):
                if p == t[k]:
                    tStr += "<b> %s (%.1f%%) </b> <br>"%(nameList[p], j[p]*100.)
                    if n==0: right1 += 1
                    right5 += 1
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
    fOutput.write(hstr.encode("utf-8"))
    if fOutput: fOutput.close()
    print "Right1 = %d (%.1f%%), Right5 = %d (%.1f%%)"%(right1, right1*100./rightTot, right5, right5*100./rightTot)

if __name__=="__main__":
    #CifarEval(fileName="Output/L4/mlp_22.hdf5")
    CifarEval(fileName="Output/FullDataset_NoAug2/mlp_5.hdf5")
