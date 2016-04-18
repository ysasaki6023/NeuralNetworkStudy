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

#dirName = "output/Conv3_Linear3_DropOut_Small"
inputPixels = 100
UseFrac  = 0.05

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
i_all,_ = np.split(Perm,[int(len(Perm)*(UseFrac))])

xp = np

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


class MLP(chainer.Chain):
    def __init__(self,N_l1 = 100, N_l2 = 100, N_l3 = 100, DropRatio = 0.5):
        super(MLP,self).__init__(
            c1=F.Convolution2D(3 , 32, 3, pad=1), # color, features, filter size
            c2=F.Convolution2D(32, 32, 3, pad=1), # color, features, filter size
            l1=F.Linear(32*10*10, N_l1),
            l2=F.Linear(N_l1, N_l2),
            l3=F.Linear(N_l2, N_l3),
            l4=F.Linear(N_l3, len(y_uniq)))
        self.DropRatio = DropRatio
        self.Train = True
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


from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
class NNClass(BaseEstimator, ClassifierMixin):
    def __init__(self,N_l1=1000,N_l2=100,DropRatio=0.5,MaxIter=20):
        self.N_l1 = N_l1
        self.N_l2 = N_l2
        self.N_l3 = N_l1
        self.DropRatio = DropRatio
        self.MinIter = 0
        self.MaxIter = MaxIter

    def fit(self,X,y):
        x_files = X
        y_data  = y

        # Setup optimizer
        self.model     = Classifier(MLP(N_l1=self.N_l1,
                                        N_l2=self.N_l2,
                                        N_l3=self.N_l3,
                                        DropRatio=self.DropRatio))
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        epoch = 0
        while True:
            epoch += 1
            print 'epoch', epoch 

            # training
            perm = np.random.permutation(len(x_files))
            sum_accuracy = 0
            sum_loss = 0
            sum_totl = 0
            time_load = 0.
            time_calc = 0.
            for i in range(0, len(x_files), batchsize):
                x = Variable(loadImages(x_files[perm[i:i + batchsize]]))
                t = Variable(xp.asarray(y_data [perm[i:i + batchsize]]))

                self.model.Train = True
                self.optimizer.zero_grads()
                loss = self.model(x,t)

                loss.backward()
                self.optimizer.update()

                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(self.model.accuracy.data) * batchsize
                sum_totl     += batchsize

                print "Training (batch: %10d/%10d)"%(i,len(y_data))
            print 'train mean loss=%3.2e, accuracy = %d%%'%( sum_loss / sum_totl, sum_accuracy / sum_totl * 100.)

            if epoch<=self.MinIter: continue
            if epoch>=self.MaxIter: break
            #if min(LossList)<min(LossList[-self.StopNumOfDown:]): break

        return self

    def score(self,X,y):
        x_files = X
        y_data  = y

        # training
        perm = np.random.permutation(len(x_files))
        y = np.zeros(len(x_files),dtype=np.int32)
        sum_loss = 0
        sum_totl = 0
        for i in range(0, len(x_files), batchsize):
            x = Variable(loadImages(x_files[perm[i:i + batchsize]]),volatile="on")
            t = Variable(xp.asarray(y_data [perm[i:i + batchsize]]),volatile="on")
            self.model.Train = False
            loss = self.model(x,t)
            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_totl     += batchsize
        return (sum_loss/sum_totl)

if __name__=="__main__":
    from sklearn import grid_search
    parameters = {'MaxIter':[10,15,20,30],
                  'N_l1'   :[100,500,1000,2000],
                  'N_l2'   :[10,50,100,500],
                  'DropRatio':[0.5],
                  }
    clf = grid_search.GridSearchCV(NNClass(),parameters,cv=5,verbose=3,n_jobs=-1)
    print "Scan over %d samples"%len(x_files[i_all])
    clf.fit(x_files[i_all],y_data[i_all])
    print "\n+ ベストパラメータ:\n"
    print clf.best_estimator_
    print"\n+ トレーニングデータでCVした時の平均スコア:\n"
    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)
