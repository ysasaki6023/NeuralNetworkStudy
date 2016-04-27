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
                 P1C_filter  = 3,
                 P1C_feature = 32,
                 P1P_ksize  = 2,
                 P1N_Normalize  = False,
                 P2C_filter  = 3,
                 P2C_feature = 32,
                 P2P_ksize  = 2,
                 P2N_Normalize  = False,
                 P3C_filter  = 3,
                 P3C_feature = 32,
                 P3N_Normalize  = False,
                 P3P_ksize  = 2,
                 L1_dropout  = 0.0,
                 L2_unit     = 256,
                 L2_dropout  = 0.0):
        super(ImageProcessNetwork, self).__init__()
        self.IsTrain = False
        self.P1N_Normalize = P1N_Normalize
        self.P2N_Normalize = P2N_Normalize
        self.P3N_Normalize = P3N_Normalize
        self.I_colors = I_colors
        self.I_Xunit  = I_Xunit
        self.I_Yunit  = I_Yunit
        self.P1P_ksize = P1P_ksize
        self.P2P_ksize = P2P_ksize
        self.P3P_ksize = P3P_ksize
        self.L1_dropout= L1_dropout
        self.L2_dropout= L2_dropout

        self.add_link("P1",L.Convolution2D( I_colors    ,  P1C_feature,  P1C_filter, pad=int(P1C_filter/2.)))
        self.P1C_nx = conv.get_conv_outsize(I_Xunit, P1P_ksize, P1P_ksize, 0, cover_all = True)
        self.P1C_ny = conv.get_conv_outsize(I_Yunit, P1P_ksize, P1P_ksize, 0, cover_all = True)

        self.add_link("P2",L.Convolution2D( P1C_feature ,  P2C_feature,  P2C_filter, pad=int(P2C_filter/2.)))
        self.P2C_nx = conv.get_conv_outsize(self.P1C_nx, P2P_ksize, P2P_ksize, 0, cover_all = True)
        self.P2C_ny = conv.get_conv_outsize(self.P1C_ny, P2P_ksize, P2P_ksize, 0, cover_all = True)

        self.add_link("P3",L.Convolution2D( P2C_feature ,  P3C_feature,  P3C_filter, pad=int(P3C_filter/2.)))
        self.P3C_nx = conv.get_conv_outsize(self.P2C_nx, P3P_ksize, P3P_ksize, 0, cover_all = True)
        self.P3C_ny = conv.get_conv_outsize(self.P2C_ny, P3P_ksize, P3P_ksize, 0, cover_all = True)

        self.add_link("L1",L.Linear( self.P3C_nx * self.P3C_ny * P3C_feature , L2_unit))
        self.add_link("L2",L.Linear( L2_unit,  F_unit))
        return

    def setTrainMode(self, IsTrain):
        self.IsTrain = IsTrain

    def __call__(self, x):
        h_P1 = self.P1(x)
        if self.P1N_Normalize: h_P1 = F.local_response_normalization(h_P1)
        h_P1 = F.max_pooling_2d(F.relu(h_P1), ksize=self.P1P_ksize, cover_all=True)

        h_P2 = self.P2(h_P1)
        if self.P2N_Normalize: h_P2 = F.local_response_normalization(h_P2)
        h_P2 = F.max_pooling_2d(F.relu(h_P2), ksize=self.P2P_ksize, cover_all=True)

        h_P3 = self.P3(h_P2)
        if self.P3N_Normalize: h_P3 = F.local_response_normalization(h_P3)
        h_P3 = F.max_pooling_2d(F.relu(h_P3), ksize=self.P3P_ksize, cover_all=True)

        h_L1 = F.dropout(F.relu(self.L1(h_P3)),ratio=self.L1_dropout,train=self.IsTrain)
        h_L2 = F.dropout(F.relu(self.L2(h_L1)),ratio=self.L2_dropout,train=self.IsTrain)
        y    = h_L2
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
    InDataBatch = []

    data_tr  = np.zeros((50000,3*32*32),dtype=np.float32)
    data_ev  = np.zeros((10000,3*32*32),dtype=np.float32)
    label_tr = np.zeros((50000),dtype=np.int32)
    label_ev = np.zeros((10000),dtype=np.int32)

    for i in range(1,5+1):
        with open("data_cifar10/data_batch_%d"%i,"r") as f:
            tmp = pickle.load(f)
            data_tr [(i-1)*10000:i*10000] = tmp["data"]
            label_tr[(i-1)*10000:i*10000] = tmp["labels"]
    with open("data_cifar10/test_batch","r") as f:
        tmp = pickle.load(f)
        data_ev  [:] = tmp["data"]
        label_ev [:] = tmp["labels"]

## Prep
    print "Normalizing data ..."
    def Normalize(x):
        avg  = np.average(x,axis=1).reshape((len(x),1))
        std  = np.sqrt(np.sum(x*x,axis=1) - np.sum(x,axis=1)).reshape((len(x),1))
        y    = (x - avg) / std
        return y
    data_tr = Normalize(data_tr)
    data_ev = Normalize(data_ev)
    print "done"

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

    model = L.Classifier(ImageProcessNetwork(I_colors=3, I_Xunit=32, I_Yunit=32, F_unit = 10, **kwd))
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
        print 'train mean loss=%.3f, accuracy=%.1f%%, throughput=%.0f images/sec'%(sum_loss / N_tr, sum_accuracy / N_tr * 100., throughput) 
        if fOutput: fOutput.write("%d,Train,%e,%e\n"%(epoch,sum_loss/N_tr,sum_accuracy/N_tr))

        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_ev, batchsize):
            x = chainer.Variable(xp.asarray(x_ev[i:i + batchsize]),volatile='on')
            t = chainer.Variable(xp.asarray(y_ev[i:i + batchsize]),volatile='on')
            model.predictor.setTrainMode(False)
            loss = model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)

        print 'test  mean loss=%.3f, accuracy=%.1f%%'%(sum_loss / N_ev, sum_accuracy / N_ev * 100, ) 
        if fOutput: fOutput.write("%d,Test,%e,%e\n"%(epoch,sum_loss/N_ev,sum_accuracy/N_ev))

        if folderName and (epoch%10 == 0 or epoch==n_epoch):
            # Save the model and the optimizer
            if epoch == n_epoch:
                myFname = os.path.join(folderName,'mlp_final')
            else:
                myFname = os.path.join(folderName,'mlp_%d'%n_epoch)

            #print 'save the model' 
            serializers.save_hdf5(myFname+".model", model)
            serializers.save_hdf5(myFname+".state", optimizer)

    if fOutput: fOutput.close()
    if fInfo  : fInfo.close()

## Base Case
n_epoch = 100

CifarAnalysis("Output/Feature666",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=2**6, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=2**6, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=2**6, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Filter777",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=7, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=7, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=7, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Baseline",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/NormalzeOnAll",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=True,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=True,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=True,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/NormalzeOnOnlyFirstLayer",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=True,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Filter555",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=5, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=5, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=5, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Feature333",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=2**3, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=2**3, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=2**3, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Feature444",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=2**4, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=2**4, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=2**4, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Feature654",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=2**6, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=2**5, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=2**4, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Feature643",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=2**6, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=2**4, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=2**3, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Ksize333",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=3,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=3,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=3,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/Ksize444",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=4,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=4,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=4,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/DropOutL0.5L0.5",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.5, L2_dropout=0.5, L2_unit   =250)

CifarAnalysis("Output/DropOutL0.2L0.2",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.2, L2_dropout=0.2, L2_unit   =250)

CifarAnalysis("Output/DropOutL0.5L0.0",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.5, L2_dropout=0.0, L2_unit   =250)

CifarAnalysis("Output/DropOutL0.0L0.5",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.5, L2_unit   =250)

CifarAnalysis("Output/Unit1000",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =1000)

CifarAnalysis("Output/Unit500",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =500)

CifarAnalysis("Output/Unit100",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   =100)

CifarAnalysis("Output/Unit50",
              n_epoch=n_epoch, batchsize = 1000,
              P1C_filter=3, P1C_feature=32, P1P_ksize=2,P1N_Normalize=False,
              P2C_filter=3, P2C_feature=32, P2P_ksize=2,P2N_Normalize=False,
              P3C_filter=3, P3C_feature=32, P3P_ksize=2,P3N_Normalize=False,
              L1_dropout=0.0, L2_dropout=0.0, L2_unit   = 50)
