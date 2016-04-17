#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle as pickle
import argparse
import time
import glob,sys
import os,csv

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

dirName = "output/Conv3_Linear3_DropOut_Large"
inputPixels = 100
Train_Frac = 0.8 # Frac will be used for the training
Test_Frac  = 0.2 # Frac will be used for the training

# Read arg
parser = argparse.ArgumentParser(description='XXX')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
#parser.add_argument('--epoch', '-e', default=200, type=int,
#                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_units = args.unit

print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
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
#y_data = np.zeros( (len(y_files), len(y_uniq) ) )
y_data = np.zeros( (len(y_files) ), dtype=np.int32 )
y_files= np.array(y_files,dtype="string")
x_files= np.array(x_files,dtype="string")
for i,s in enumerate(y_uniq):
    y_data[y_files == s] = i

# Prepare training data
Perm = np.random.permutation(len(y_data))
i_train,i_test,_ = np.split(Perm,[int(len(Perm)*(Train_Frac)),int(len(Perm)*(Train_Frac+Test_Frac))])

xp = np

class MLP(chainer.Chain):
    def __init__(self):
        super(MLP,self).__init__(
            c1=F.Convolution2D(3 , 32, 3, pad=1), # color, features, filter size
            c2=F.Convolution2D(32, 32, 3, pad=1), # color, features, filter size
            l1=F.Linear(32*10*10, 1000),
            l2=F.Linear(1000, 100),
            l3=F.Linear(100, 100),
            l4=F.Linear(100, len(y_uniq)))
        self.Train = True
        self.DropRatio = 0.5
    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.c1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.c2(h)), 5)
        h = F.dropout(F.relu(self.l1(h)),train = self.Train, ratio = self.DropRatio)
        h = F.dropout(F.relu(self.l2(h)),train = self.Train, ratio = self.DropRatio)
        h = F.dropout(F.relu(self.l3(h)),train = self.Train, ratio = self.DropRatio)
        y  = self.l4(h)
        return y

class Classifier(chainer.Chain):
    def __init__(self,predictor):
        super(Classifier,self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(y,t)
        self.accuracy = F.accuracy(y,t)
        self.y = y
        return self.loss


# Setup optimizer
model     = Classifier(MLP())
model.epoch = 0

# Init/Resume
if args.resume:
    print('Load model state from', args.resume)
    with open(args.resume,"rb") as f:
        model = pickle.load(f)

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
        nimg = np.array(img) / 256.
        # plt.imshow(nimg)
        # plt.show()
        #aimg = nimg.ravel() / 256.
        aimg = nimg.transpose(2,0,1)
        if iFile==0:
            x_data = np.zeros([len(fileNames)]+list(aimg.shape),dtype=np.float32)
        x_data[iFile,:] = aimg
    return x_data

def dumpFile(outFileName, dataIndex, oriClass, estClass, comment):
    with open(outFileName,"w") as f:
        writer = csv.writer(f,lineterminator='\n')
        header = ["index","originalClass","estimatedClass","flag"]
        writer.writerow(header)
        for line in zip(dataIndex,oriClass,estClass,comment):
            writer.writerow(line)

# Learning loop

while True:
    model.epoch += 1
    print 'epoch', model.epoch 

    # save
    save_dataIndex     = np.zeros(len(i_train)+len(i_test),dtype=np.int32)
    #save_fileName      = np.zeros(len(i_train)+len(i_test),dtype="string")
    save_oriClass      = np.zeros(len(i_train)+len(i_test),dtype=np.int32)
    save_estClass      = np.zeros(len(i_train)+len(i_test),dtype=np.int32)
    save_comment       = np.zeros(len(i_train)+len(i_test),dtype="string")
    save_Ntrain        = 0
    save_Ntest         = 0

    # training
    perm = np.random.permutation(len(i_train))
    sum_accuracy = 0
    sum_loss = 0
    sum_totl = 0
    time_load = 0.
    time_calc = 0.
    epoch_start = time.time()
    for i in range(0, len(i_train), batchsize):
        start = time.time()
        x = Variable(loadImages(x_files[i_train[perm[i:i + batchsize]]]))
        t = Variable(xp.asarray(y_data [i_train[perm[i:i + batchsize]]]))
        mid   = time.time()

        model.Train = True
        optimizer.zero_grads()
        loss = model(x,t)

        loss.backward()
        optimizer.update()

        end   = time.time()

        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(model.accuracy.data) * batchsize
        sum_totl     += batchsize
        time_load += mid - start
        time_calc += end - mid

        Nlen = len(i_train[perm[i:i + batchsize]])
        save_dataIndex[i:i+Nlen] = i_train[perm[i:i + batchsize]]
        #save_fileName [i:i+Nlen] = x_files[i_train[perm[i:i + batchsize]]]
        save_oriClass [i:i+Nlen] = y_data [i_train[perm[i:i + batchsize]]]
        save_estClass [i:i+Nlen] = np.argmax(model.y.data,axis=1)
        save_comment  [i:i+Nlen] = "t"
        save_Ntrain += Nlen
        print "Training (batch: %10d/%10d)"%(i,len(i_train))
    train_loss = sum_loss     / sum_totl
    train_accu = sum_accuracy / sum_totl
    print 'train mean loss=%3.2e, accuracy = %d%%'%( train_loss, train_accu * 100.)

    # test
    perm = np.random.permutation(len(i_test))
    sum_loss = 0
    sum_totl = 0
    sum_accuracy = 0
    for i in range(0, len(i_test), batchsize):
        x = Variable(loadImages(x_files[i_test[perm[i:i + batchsize]]]),volatile="on")
        t = Variable(xp.asarray(y_data [i_test[perm[i:i + batchsize]]]),volatile="on")
        model.Train = False
        loss = model(x,t)

        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(model.accuracy.data) * batchsize
        sum_totl     += batchsize

        Nlen = len(i_test[perm[i:i + batchsize]])
        save_dataIndex[save_Ntrain+i:save_Ntrain+i+Nlen] = i_test[perm[i:i + batchsize]]
        #save_fileName [save_Ntrain+i:save_Ntrain+i+Nlen] = x_files[i_test[perm[i:i + batchsize]]]
        save_oriClass [save_Ntrain+i:save_Ntrain+i+Nlen] = y_data [i_test[perm[i:i + batchsize]]]
        save_estClass [save_Ntrain+i:save_Ntrain+i+Nlen] = np.argmax(model.y.data,axis=1)
        save_comment  [save_Ntrain+i:save_Ntrain+i+Nlen] = "e"
        save_Ntest += Nlen
        print "Testing  (batch: %10d/%10d)"%(i,len(i_test))
    test_loss = sum_loss     / sum_totl
    test_accu = sum_accuracy / sum_totl
    print 'test  mean loss=%3.2e, accuracy = %d%%'%( test_loss, test_accu * 100.)

    # Save the model and the optimizer
    print 'saving files ...'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    if model.epoch == 1:
        with open(os.path.join(dirName,"classIndex.csv"),"w") as f:
            writer = csv.writer(f,lineterminator='\n')
            for line in enumerate(y_uniq):
                writer.writerow(line)
        with open(os.path.join(dirName,"fileIndex.csv"),"w") as f:
            writer = csv.writer(f,lineterminator='\n')
            for i,line in enumerate(zip(y_data,y_files,x_files)):
                writer.writerow([i]+list(line))
        addMode = "w"
    else:
        addMode = "a"
    with open(os.path.join(dirName,"trainProgress.csv"),addMode) as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow([model.epoch,train_loss,train_accu,test_loss,test_accu])

    with open(os.path.join(dirName,"model.pickle"),"wb") as f:
        pickle.dump(model,f)
                
    dumpFile(os.path.join(dirName,"epoch_%d.csv"%model.epoch),
             save_dataIndex[:save_Ntrain+save_Ntest],
             save_oriClass[:save_Ntrain+save_Ntest],
             save_estClass[:save_Ntrain+save_Ntest],
             save_comment[:save_Ntrain+save_Ntest])

    print 'done' 
    epoch_end = time.time()
    print "Time spent: %5.0fsec, Training: %3.2e events, Testing: %3.2e events"%(epoch_end-epoch_start,save_Ntrain,save_Ntest)

