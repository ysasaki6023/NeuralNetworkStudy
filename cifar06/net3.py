#!/usr/bin/env python
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.utils import conv

class ImageProcessNetwork(chainer.Chain):
    def __init__(self, F_unit):
        super(ImageProcessNetwork, self).__init__()
        self.IsTrain = True
        self.F_unit = F_unit

        self.add_link("P1_1",L.Convolution2D(  3,    96, 3, pad=1 ))
        self.add_link("BN1_1",L.BatchNormalization(96))
        self.add_link("P1_2",L.Convolution2D( 96,    96, 3, pad=1 ))
        self.add_link("BN1_2",L.BatchNormalization(96))
        self.add_link("P2_1",L.Convolution2D( 96,   192, 3, pad=1 ))
        self.add_link("P2_2",L.Convolution2D(192,   192, 3, pad=1 ))
        self.add_link("P2_3",L.Convolution2D(192,   192, 3, pad=1 ))
        self.add_link("P3_1",L.Convolution2D(192,   192, 3, pad=1 ))
        self.add_link("P3_2",L.Convolution2D(192,   192, 1, pad=1 ))
        self.add_link("P3_3",L.Convolution2D(192,F_unit, 1, pad=1 ))
        return

    def setTrainMode(self, IsTrain):
        self.IsTrain = IsTrain
        return

    def __call__(self, x):
        h = x
        h = self.__dict__["P1_1"](F.elu(h))
        h = self.__dict__["BN1_1"](h)
        h = self.__dict__["P1_2"](F.elu(h))
        h = self.__dict__["BN1_2"](h)
        h = F.max_pooling_2d(F.elu(h), ksize=3, stride=2, cover_all=False)
        h = self.__dict__["P2_1"](h)
        h = self.__dict__["P2_2"](F.elu(h))
        h = self.__dict__["P2_2"](F.elu(h))
        h = F.max_pooling_2d(F.elu(h), ksize=3, stride=2, cover_all=False)
        h = self.__dict__["P3_1"](h)
        h = self.__dict__["P3_2"](F.elu(h))
        h = self.__dict__["P3_3"](F.elu(h))
        h = F.average_pooling_2d(F.elu(h), ksize=6)
        y = F.reshape(h,(len(h.data),self.F_unit))
        return y

def GenModel(F_unit):
    return L.Classifier(ImageProcessNetwork(F_unit))
