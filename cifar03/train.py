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


def CifarAnalysis(folderName=None,n_epoch=1,batchsize = 1000, **kwd):
    id_gpu  = 0

    OutStr = ""
    OutStr += 'GPU: {}\n'.format(id_gpu)
    OutStr += 'Minibatch-size: {}\n'.format(batchsize) 
    OutStr += 'epoch: {}\n'.format(n_epoch) 
    OutStr += 'kwd: {}\n'.format(kwd) 
    OutStr += '' 

    print OutStr

    fOutput = None
    fInfo   = None
    if folderName:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        fOutput = open(os.path.join(folderName,"output.dat"),"w")
        fInfo   = open(os.path.join(folderName,"info.dat"),"w")
        shutil.copyfile(__file__,os.path.join(folderName,os.path.basename(__file__)))

    if fInfo: fInfo.write(OutStr)

# Prepare dataset
    data_tr  = np.zeros((50000,3*32*32),dtype=np.float32)
    data_ev  = np.zeros((10000,3*32*32),dtype=np.float32)
    label_tr = np.zeros((50000),dtype=np.int32)
    label_ev = np.zeros((10000),dtype=np.int32)
    I_colors = 3
    I_Xunit  = 32
    I_Yunit  = 32
    F_unit   = 100   # be careful!!

    with open("data_cifar100/train.pickle","r") as f:
        tmp = pickle.load(f)
        data_tr  [:] = tmp["data"]
        label_tr [:] = np.array(tmp["fine_labels"],dtype=np.int32)
    with open("data_cifar100/test.pickle","r") as f:
        tmp = pickle.load(f)
        data_ev  [:] = tmp["data"]
        label_ev [:] = np.array(tmp["fine_labels"],dtype=np.int32)

## Prep
    print "Normalizing data ..."
    def Normalize(x):
        avg  = np.average(x,axis=1).reshape((len(x),1))
        std  = np.sqrt(np.sum(x*x,axis=1) - np.sum(x,axis=1)).reshape((len(x),1))
        y    = (x - avg) / std
        return y
    data_tr = Normalize(data_tr)
    data_ev = Normalize(data_ev)

    x_tr = data_tr.reshape((len(data_tr),3,32,32))
    x_ev = data_ev.reshape((len(data_ev),3,32,32))
    y_tr = label_tr
    y_ev = label_ev
    N_tr = len(data_tr) # 50000
    N_ev = len(data_ev) # 10000

## Define analisis
    Resume = None
    if "Resume" in kwd:
        Resume = kwd["Resume"]
        del kwd["Resume"]

    model = L.Classifier(ImageProcessNetwork(I_colors=I_colors, I_Xunit=I_Xunit, I_Yunit=I_Yunit, F_unit = F_unit, **kwd))
    if id_gpu >= 0:
        cuda.get_device(id_gpu).use()
        model.to_gpu()
    xp = np if id_gpu < 0 else cuda.cupy

# Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

# Init/Resume
    if Resume:
        print('Load optimizer state from', Resume)
        serializers.load_hdf5(Resume+".state", optimizer)
        serializers.load_hdf5(Resume+".model", model)

# Learning loop
    if fOutput: fOutput.write("epoch,mode,loss,accuracy\n")
    for epoch in six.moves.range(1, n_epoch + 1):
        print 'epoch %d'%epoch 

        # training
        perm = np.random.permutation(N_tr)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0, N_tr, batchsize):
            x = chainer.Variable(xp.asarray(x_tr[perm[i:i + batchsize]]))
            t = chainer.Variable(xp.asarray(y_tr[perm[i:i + batchsize]]))

            # Pass the loss function (Classifier defines it) and its arguments
            model.predictor.setTrainMode(True)
            optimizer.update(model, x, t)

            if (epoch == 1 and i == 0) and folderName:
                with open(os.path.join(folderName,'graph.dot'), 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (model.loss, ))
                    o.write(g.dump())
                print 'graph generated' 

            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
        end = time.time()
        elapsed_time = end - start
        throughput = N_tr / elapsed_time
        print 'train mean loss=%.5f, accuracy=%.2f%%, throughput=%.0f images/sec'%(sum_loss / N_tr, sum_accuracy / N_tr * 100., throughput) 
        if fOutput: fOutput.write("%d,Train,%e,%e\n"%(epoch,sum_loss/N_tr,sum_accuracy/N_tr))

        # evaluation
        perm = np.random.permutation(N_ev)
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_ev, batchsize):
            x = chainer.Variable(xp.asarray(x_ev[perm[i:i + batchsize]]),volatile='on')
            t = chainer.Variable(xp.asarray(y_ev[perm[i:i + batchsize]]),volatile='on')
            model.predictor.setTrainMode(False)
            loss = model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)

        print 'test  mean loss=%.5f, accuracy=%.2f%%'%(sum_loss / N_ev, sum_accuracy / N_ev * 100, ) 
        if fOutput: fOutput.write("%d,Test,%e,%e\n"%(epoch,sum_loss/N_ev,sum_accuracy/N_ev))

        if folderName and (epoch%10 == 0 or epoch==n_epoch):
            # Save the model and the optimizer
            if epoch == n_epoch:
                myFname = os.path.join(folderName,'mlp_final')
            else:
                myFname = os.path.join(folderName,'mlp_%d'%epoch)

            #print 'save the model' 
            serializers.save_hdf5(myFname+".model", model)
            serializers.save_hdf5(myFname+".state", optimizer)
            with open(myFname+".pickle","wb") as f:
                pickle.dump(optimizer,f)

    if fOutput: fOutput.close()
    if fInfo  : fInfo.close()

if __name__=="__main__":
    n_epoch = 2000
    CifarAnalysis("Output/L4test",
                   n_epoch=n_epoch,
                   batchsize = 1000,
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
                   L2_unit     = 500)
