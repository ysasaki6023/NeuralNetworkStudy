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

import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm


class ImageProcessNetwork(chainer.Chain):
    def __init__(self,
                 I_colors, I_Xunit, I_Yunit, F_unit,
                   N_PLayers = 4,
                   P0C_feature = 32,
                   P1C_feature = 32,
                   P2C_feature = 16,
                   P0C_filter  = 3,
                   P1C_filter  = 3,
                   P2C_filter  = 3,
                   P0P_ksize   = 2,
                   P1P_ksize   = 2,
                   P2P_ksize   = 2,
                   L1_dropout  = 0.5,
                   L2_dropout  = 0.0,
                   L2_unit     = 500):
        super(ImageProcessNetwork, self).__init__()
        self.IsTrain = True

        self.NPLayers  = N_PLayers
        self.NFeatures = [I_colors]
        self.NFilter   = [1]
        self.NKsize    = [1]
        self.NImgPix   = [(I_Xunit,I_Yunit)]
        self.L1_dropout = L1_dropout
        self.L2_dropout = L2_dropout
        self.L2_unit    = L2_unit
        for iL in range(self.NPLayers):
            ## Set Variables
            self.NFeatures.append(self.gradualVariable(iL,self.NPLayers,P0C_feature,P1C_feature,P2C_feature))
            self.NFilter.append(  self.gradualVariable(iL,self.NPLayers,P0C_filter ,P1C_filter ,P2C_filter ))
            self.NKsize.append(   self.gradualVariable(iL,self.NPLayers,P0P_ksize  ,P1P_ksize  ,P2P_ksize  ))
            ## Update layers
            self.NImgPix.append(
                ( conv.get_conv_outsize( self.NImgPix[-1][0], self.NKsize[-1], self.NKsize[-1], 0, cover_all = True),
                  conv.get_conv_outsize( self.NImgPix[-1][1], self.NKsize[-1], self.NKsize[-1], 0, cover_all = True)))
            self.add_link("P%d"%iL,L.Convolution2D( self.NFeatures[-2], self.NFeatures[-1],
                                                    self.NFilter[-1]  , pad=int(self.NFilter[-1]/2.)))

        self.add_link("L1",L.Linear( self.NImgPix[-1][0] * self.NImgPix[-1][1] * self.NFeatures[-1] , L2_unit))
        self.add_link("L2",L.Linear( L2_unit,  F_unit))

        return

    def gradualVariable(self, cLayer, tLayer, val0, val1, val2):
        pos = 0.5
        if cLayer <= int(pos*tLayer): v0, v1, p0, p1, pc = val0, val1, 0, int(pos*tLayer), int( cLayer - 0 )
        else                        : v0, v1, p0, p1, pc = val1, val2, int(pos*tLayer), tLayer-1, int( cLayer - int(pos*tLayer))
        return int(float(v0) + (float(v1)-float(v0))/(float(p1)-float(p0))*float(pc))

    def setTrainMode(self, IsTrain):
        self.IsTrain = IsTrain
        return

    def __call__(self, x):
        h = x
        for iL in range(self.NPLayers):
            h = self.__dict__["P%d"%iL](h)
            h = F.local_response_normalization(h)
            h = F.max_pooling_2d(F.relu(h), ksize=self.NKsize[iL+1], cover_all=True)

        h = F.dropout(F.relu(self.L1(h)),ratio=self.L1_dropout,train=self.IsTrain)
        h = F.dropout(F.relu(self.L2(h)),ratio=self.L2_dropout,train=self.IsTrain)
        y    = h
        return y


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

    with open("data_cifar100/test.pickle","rb") as f:
        tmp = pickle.load(f)
        data_ev  [:] = tmp["data"]
        label_ev [:] = np.array(tmp["fine_labels"],dtype=np.int32)
    with open("data_cifar100/batches.meta","rb") as f:
        tmp = pickle.load(f)
        print tmp.keys()
        nameList  = tmp["fine_label_names"]
        #nameList [:] = tmp["fine_label_names"]

## Prep
    print "Normalizing data ..."
    def Normalize(x):
        avg  = np.average(x,axis=1).reshape((len(x),1))
        std  = np.sqrt(np.sum(x*x,axis=1) - np.sum(x,axis=1)).reshape((len(x),1))
        y    = (x - avg) / std
        return y
    data_ev = Normalize(data_ev)

    x_ev = data_ev.reshape((len(data_ev),3,32,32))
    y_ev = label_ev
    N_ev = len(data_ev) # 10000

    model = L.Classifier(ImageProcessNetwork(I_colors=I_colors, I_Xunit=I_Xunit, I_Yunit=I_Yunit, F_unit = F_unit))
    if id_gpu >= 0:
        cuda.get_device(id_gpu).use()
        model.to_gpu()
    xp = np if id_gpu < 0 else cuda.cupy

# Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

# Init/Resume
    print 'Load optimizer state from %s'%fInput
    serializers.load_hdf5(fInput+".state", optimizer)
    serializers.load_hdf5(fInput+".model",model)
    model.predictor.setTrainMode(False)

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
    CifarEval(fileName="Output/L4test/mlp_final")
