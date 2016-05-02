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

# Prepare dataset

data_tr  = np.zeros((50000,3*32*32),dtype=np.float32)
data_ev  = np.zeros((10000,3*32*32),dtype=np.float32)
label_tr = np.zeros((50000),dtype=np.int32)
label_ev = np.zeros((10000),dtype=np.int32)
label_name = []

with open("data_cifar100/train.pickle","rb") as f:
    tmp = pickle.load(f)
    data_tr  [:] = tmp["data"]
    label_tr [:] = np.array(tmp["coarse_labels"],dtype=np.int32)
with open("data_cifar100/test.pickle","rb") as f:
    tmp = pickle.load(f)
    data_ev  [:] = tmp["data"]
    label_ev [:] = np.array(tmp["coarse_labels"],dtype=np.int32)
with open("data_cifar100/batches.meta","rb") as f:
    tmp = pickle.load(f)
#['fine_label_names', 'coarse_label_names']
    label_name = tmp["coarse_label_names"]
    #label_name = tmp["label_names"]

## Prep
print "Normalizing data ..."
def Normalize(x):
    avg  = np.average(x,axis=1).reshape((len(x),1))
    std  = np.sqrt(np.sum(x*x,axis=1) - np.sum(x,axis=1)).reshape((len(x),1))
    y    = (x - avg) / std
    return y
#data_tr = Normalize(data_tr)
#data_ev = Normalize(data_ev)

x_tr = data_tr.reshape((len(data_tr),3,32,32))
x_ev = data_ev.reshape((len(data_ev),3,32,32))
y_tr = label_tr
y_ev = label_ev
N_tr = len(data_tr) # 50000
N_ev = len(data_ev) # 10000

# print
tx,ty = x_tr,y_tr
for i in range(10):
    print ty[i], label_name[ty[i]]
    #plt.imshow(data_tr[i].reshape(3,32,32).transpose(1,2,0)/255.)
    plt.imshow(tx[i].transpose(1,2,0))
    plt.show()
