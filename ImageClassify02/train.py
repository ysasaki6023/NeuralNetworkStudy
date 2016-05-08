#!/usr/bin/env python
import argparse
import time

import numpy as np
import six
import os
import shutil
import imp
import os

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
import StopManager
import h5py
import net
netFile = os.path.join(imp.find_module("net")[0]).read() # To save the current net structure in the hdf5 file


def CifarAnalysis(folderName=None,n_epoch = 8,batchsize = 1000, **kwd):
    id_gpu  = 0

    OutStr = ""
    OutStr += 'GPU: {}\n'.format(id_gpu)
    OutStr += 'Minibatch-size: {}\n'.format(batchsize) 
    OutStr += 'kwd: {}\n'.format(kwd) 
    OutStr += '' 
    print OutStr


    fOutput = None
    if folderName:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        fOutput = open(os.path.join(folderName,"output.dat"),"w")
        shutil.copyfile(__file__,os.path.join(folderName,os.path.basename(__file__)))


# Prepare dataset
    h5f = h5py.File("input.dat","r")
    data     = h5f["Original/Data"].value
    label    = h5f["Info/Label"].value
    imIndex  = h5f["Info/ImIndex"].value
    agIndex  = h5f["Info/AgIndex"].value

    TraEva   = np.random.random(max(imIndex)+1).repeat(3)
    TraEva   = TraEva<0.8
    with h5py.File(os.path.join(folderName,"TraEva.hdf5"),"w") as af:
        af.create_dataset("TraEva",data=TraEva,dtype=TraEva.dtype)
        af.flush()
    data_tr  = data[TraEva]
    label_tr = label[TraEva]
    data_ev  = data[np.logical_not(TraEva)]
    label_ev = label[np.logical_not(TraEva)]
    agIndex_ev = agIndex[np.logical_not(TraEva)]
    data_ev  = data_ev [agIndex_ev == 0]
    label_ev = label_ev[agIndex_ev == 0]

    I_colors = 3
    I_Xunit  = 32
    I_Yunit  = 32
    F_unit   = max(label)+1   # be careful!!

    #print np.mean(data_tr,axis=1).reshape(3*32*32)
    data_tr -= np.mean(data_tr,axis=1).reshape((len(data_tr),1))
    data_tr -= np.mean(data_tr,axis=0)
    data_tr /= np.std (data_tr,axis=0)

    data_ev -= np.mean(data_ev,axis=1).reshape((len(data_ev),1))
    data_ev -= np.mean(data_ev,axis=0)
    data_ev /= np.std (data_ev,axis=0)

## Prep
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

    #model,ModelKwd = net2.GenModel(F_unit = F_unit)
    model = net.GenModel(F_unit)
    ModelKwd = ""
    if id_gpu >= 0:
        cuda.get_device(id_gpu).use()
        model.to_gpu()
    xp = np if id_gpu < 0 else cuda.cupy

# Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    #optimizer.add_hook(scheduled_alpha_reduction)

# Init/Resume
    if Resume:
        print 'Load optimizer state from %s'%(Resume)
        with h5py.File(Resume,"r") as f:
            s = HDF5Deserializer(f)

            s_model = s["model"]
            s_model.load(model)

# Setup stop manager
    sm = StopManager.StopManager()
    sm.SetMaximumEpoch(10000)
    sm.SetMinimumEpoch(10)
    sm.SetStopThreshold(3e-4)
    print sm

    #alphaTiming = [10,20,40,80]
    #optimizer.alpha /= 16

# Learning loop
    if fOutput: fOutput.write("epoch,mode,loss,accuracy\n")
    #for epoch in six.moves.range(1, n_epoch + 1):
    epoch = 0
    while True:
        epoch += 1
        print 'epoch %d'%epoch 

        # training
        perm = np.random.permutation(N_tr)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        """
        if epoch in alphaTiming:
            optimizer.alpha /= 2
            print "alpha changed... currently Alpha = ", optimizer.alpha
        """
        for i in six.moves.range(0, N_tr, batchsize):
            bx = x_tr[perm[i:i + batchsize]]
            #bx = ag.Aug(bx)
            x = chainer.Variable(xp.asarray(bx))
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
        sm.AddAccuracy(sum_accuracy/N_ev)
        print sm.GetInfo()
        if fOutput: fOutput.write("%d,Test,%e,%e\n"%(epoch,sum_loss/N_ev,sum_accuracy/N_ev))

        StopFlag = sm.StopCheck()
        if epoch>=n_epoch: StopFlag = True
        #StopFlag = False

        if folderName and (epoch%1 == 0 or StopFlag):
            # Save the model and the optimizer
            if StopFlag:
                myFname = os.path.join(folderName,'mlp_final')
            else:
                myFname = os.path.join(folderName,'mlp_%d'%epoch)

            with h5py.File(myFname+".hdf5","w") as f:
                s = HDF5Serializer(f)
                s["model"].save(model)
                f.create_dataset("kwd",data=ModelKwd.__str__(),dtype=h5py.special_dtype(vlen=unicode))
                f.create_dataset("net",data=netFile,dtype=h5py.special_dtype(vlen=unicode))
                f.flush()

        if StopFlag: break

    if fOutput: fOutput.close()

if __name__=="__main__":
    CifarAnalysis("Output/FullDataset_NoAug2", batchsize = 250, n_epoch=8)
