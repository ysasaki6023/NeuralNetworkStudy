#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle as pickle
import argparse
import time
import glob,sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Variable, FunctionSet
from chainer import serializers

# Set variables
dataFolder = "data"
inputPixels = 10

# Read arg
parser = argparse.ArgumentParser(description='XXX')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--epoch', '-e', default=200, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=2000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
n_units = args.unit

print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# define input samples

x_files = []
y_files = []

for fname in glob.glob(dataFolder+"/*/*.jpg"):
    #print fname
    _,y,x = fname.replace(".jpg","").split("/")
    x_files.append(fname)
    y_files.append(y)
 
y_uniq = list(set(y_files))
y_data = np.zeros( (len(y_files), len(y_uniq) ) )
y_files= np.array(y_files,dtype="string")
x_files= np.array(x_files,dtype="string")
for i,s in enumerate(y_uniq):
    tmp = np.zeros( len(y_files) )
    tmp[ y_files == s ] = 1
    y_data[:,i] = tmp 
y_data = y_data.astype(np.float32)

# Prepare training data
Frac = 0.9 # Frac will be used for the training
Perm = np.random.permutation(len(y_data))
i_train,i_test = np.split(Perm,[int(len(Perm)*Frac)])

xp = np

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

model = FunctionSet(l1=F.Linear(inputPixels*inputPixels*3, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, len(y_uniq)))

def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x )), train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y  = model.l3(h2)
    return F.mean_squared_error(y, t), y

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)


def loadImages(fileNames):
    x_data = None
    for iFile,fName in enumerate(fileNames):
        img = Image.open( fName ).convert("RGB")
        size = img.size
        cropBox = (int( size[0]/2 - min(size)/2), int( size[1]/2 - min(size)/2),
                   int( size[0]/2 + min(size)/2), int( size[1]/2 + min(size)/2))
        img = img.crop(cropBox)
        img = img.resize((inputPixels,inputPixels))
        nimg = np.array(img)
        # plt.imshow(nimg)
        # plt.show()
        aimg = nimg.ravel()
        if iFile==0:
            x_data = np.zeros((len(fileNames),len(aimg)),dtype=np.float32)
        x_data[iFile,:] = aimg
    return x_data

# Learning loop
for epoch in range(1, n_epoch + 1):
    print 'epoch', epoch 
    # training
    perm = np.random.permutation(len(i_train))
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in range(0, len(i_train), batchsize):
        # load images
        x = loadImages(x_files[i_train[perm[i:i + batchsize]]])
        t = xp.asarray(y_data [i_train[perm[i:i + batchsize]]])
        optimizer.zero_grads()
        loss, prod = forward(x,t)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        print i

    print ('train mean loss={}'.format(sum_loss / N))
    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)

