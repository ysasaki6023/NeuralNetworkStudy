#!/usr/bin/env python
import argparse
import time

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

import cPickle as pickle


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
                 L2_dropout  = 0.5):
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
        self.P1C_nx = int( (I_Xunit+1) / P1P_ksize)
        self.P1C_ny = int( (I_Yunit+1) / P1P_ksize)
        self.P1P_pad = (I_Xunit - self.P1C_nx * P1P_ksize,I_Yunit - self.P1C_ny * P1P_ksize)

        self.add_link("P2",L.Convolution2D( P1C_feature ,  P2C_feature,  P2C_filter, pad=int(P2C_filter/2.)))
        self.P2C_nx = int( (self.P1C_nx+1) / P2P_ksize)
        self.P2C_ny = int( (self.P1C_ny+1) / P2P_ksize)
        self.P2P_pad = (self.P1C_nx - self.P2C_nx * P2P_ksize, self.P1C_ny - self.P2C_ny * P2P_ksize)

        self.add_link("P3",L.Convolution2D( P2C_feature ,  P3C_feature,  P3C_filter, pad=int(P3C_filter/2.)))
        self.P3C_nx = int( (self.P2C_nx+1) / P3P_ksize)
        self.P3C_ny = int( (self.P2C_ny+1) / P3P_ksize)
        self.P3P_pad = (self.P2C_nx - self.P3C_nx * P3P_ksize, self.P2C_ny - self.P3C_ny * P3P_ksize)

        self.add_link("L1",L.Linear( self.P3C_nx * self.P3C_ny * P3C_feature , L2_unit))
        self.add_link("L2",L.Linear( L2_unit,  F_unit))
        return

    def setTrainMode(self, IsTrain):
        self.IsTrain = IsTrain

    def __call__(self, x):
        h_P1 = self.P1(x)
        if self.P1N_Normalize: h_P1 = F.local_response_normalization(h_P1)
        h_P1 = F.max_pooling_2d(F.relu(h_P1), ksize=self.P1P_ksize, pad=self.P1P_pad)

        h_P2 = self.P2(h_P1)
        if self.P2N_Normalize: h_P2 = F.local_response_normalization(h_P2)
        h_P2 = F.max_pooling_2d(F.relu(h_P2), ksize=self.P2P_ksize, pad=self.P2P_pad)

        h_P3 = self.P3(h_P2)
        if self.P3N_Normalize: h_P3 = F.local_response_normalization(h_P3)
        h_P3 = F.max_pooling_2d(F.relu(h_P3), ksize=self.P3P_ksize, pad=self.P3P_pad)

        h_L1 = F.dropout(F.relu(self.L1(h_P3)),ratio=self.L1_dropout,train=self.IsTrain)
        h_L2 = F.dropout(F.relu(self.L2(h_L1)),ratio=self.L2_dropout,train=self.IsTrain)
        y    = h_L2
        return y


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=200, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=1000,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch

print 'GPU: {}'.format(args.gpu)
print '# Minibatch-size: {}'.format(args.batchsize) 
print '# epoch: {}'.format(args.epoch) 
print '' 

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

data_tr = data_tr.reshape((len(data_tr),3,32,32))
data_ev = data_ev.reshape((len(data_ev),3,32,32))

## Prep
# ...
# ...
x_tr = data_tr / 256.
x_ev = data_ev / 256.
y_tr = label_tr
y_ev = label_ev
N_tr = len(data_tr) # 50000
N_ev = len(data_ev) # 10000

model = L.Classifier(ImageProcessNetwork(I_colors=3, I_Xunit=32, I_Yunit=32, F_unit = 10))
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
f = open("output.dat","w")
f.write("epoch,mode,accuracy\n")
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

        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ))
                o.write(g.dump())
            print 'graph generated' 

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N_tr / elapsed_time
    print 'train mean loss=%.3f, accuracy=%.1f%%, throughput=%.0f images/sec'%(
        sum_loss / N_tr, sum_accuracy / N_tr * 100., throughput) 
    f.write("%d,Train,%e\n"%(epoch,sum_accuracy/N_tr))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, 10000, batchsize):
        x = chainer.Variable(xp.asarray(x_ev[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_ev[i:i + batchsize]),
                             volatile='on')
        model.predictor.setTrainMode(False)
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print 'test  mean loss=%.3f, accuracy=%.1f%%'%(
        sum_loss / N_ev, sum_accuracy / N_ev * 100, ) 
    f.write("%d,Test,%e\n"%(epoch,sum_accuracy/N_ev))

f.close()

# Save the model and the optimizer
print 'save the model' 
serializers.save_npz('mlp.model', model)
print 'save the optimizer' 
serializers.save_npz('mlp.state', optimizer)
