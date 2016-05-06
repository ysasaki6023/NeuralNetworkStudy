#!/usr/bin/env python
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.utils import conv

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

def GenModel(**kwd):
    return L.Classifier(ImageProcessNetwork(**kwd)),kwd
