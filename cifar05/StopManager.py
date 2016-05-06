import numpy as np
import pandas as pd
import time

class StopManager:
    def __init__(self):
        self.Accuracy = []
        self.Time     = []
        self.Epoch    = []
        self.MaxEpoch = None
        self.MinEpoch = 10
        self.LookBackRatioForP1 = 0.2 # Look back 20% of epoch to calculate P1
        self.AverageNum = 10
        self.Threshold = 3e-4
        return

    def __str__(self):
        return "StopManager: Threshold = %.1f%% / 100epoch, MinEpoch = %d, MaxEpoch = %d" % (self.Threshold*100*100,self.MinEpoch,self.MaxEpoch)

    def GetInfo(self):
        params = self.GetAI()
        if np.isnan(params["AIrate"]):
            return ""
        else:
            return ("Current:      Accuracy = %.1f%%, Epoch = %.0f, Improvement = +%.1f%% / 100 epoch ( = %.0f min ), Stop threshold = %.1f%% / 100epoch\n"%(params["Current:Accuracy"]*100,params["Current:Epoch"],params["AIrate"]*100*100,params["EpochTime"]*100/60.,self.Threshold*100*100)
        +"At threshold: Accuracy = %.1f%%, Epoch = %.0f, Time remaining = %.0f min"%(params["Threshold:Accuracy"]*100.,params["Threshold:Epoch"],params["Threshold:TimeRemaining"]/60.))


    def SetMaximumEpoch(self,maxEpoch=None):
        self.MaxEpoch = maxEpoch
        return

    def SetMinimumEpoch(self,minEpoch=10):
        self.MinEpoch = minEpoch
        return

    def SetStopThreshold(self,threshold=3e-4):
        self.Threshold = threshold
        return

    def AddAccuracy(self,accuracy):
        self.Accuracy.append(accuracy)
        self.Time.append(time.time())
        self.Epoch.append(len(self.Epoch)+1)
        return

    def GetAI(self):
        epoch  = np.array(self.Epoch,dtype=np.int32)
        accur  = np.array(self.Accuracy,dtype=np.float32)
        deltaE = self.LookBackRatioForP1
        p1    = (accur-accur[(epoch*(1-deltaE)).astype(np.int)])/(np.log(epoch)-np.log(epoch*(1-deltaE)))
        p1avg = np.array(pd.Series(p1).rolling(window=self.AverageNum).mean())
        ai    = p1 / epoch
        aiavg = p1avg / epoch
        atime = np.array(self.Time,dtype=np.float64)
        atime -= atime[0]
        timeAvg = (atime[-1] - atime[(epoch[-1]*(1-deltaE)).astype(np.int)]) / (epoch[-1] - (epoch[-1]*(1-deltaE)).astype(np.int))

        params = {}
        Et = p1[-1] / self.Threshold
        params["Threshold:TimeRemaining"] = timeAvg * p1[-1] * (1./self.Threshold - 1./aiavg[-1])
        params["Threshold:Epoch"]         = Et
        params["Threshold:Accuracy"]      = accur[-1] + p1[-1] * (np.log(Et) - np.log(epoch[-1]))
        params["Current:Epoch"]           = epoch[-1]
        params["Current:Accuracy"]        = accur[-1]
        params["AIrate"]                  = aiavg[-1]
        params["EpochTime"]               = timeAvg
        return params

    def StopCheck(self):
        epoch = len(self.Accuracy)
        if self.MaxEpoch and epoch >= self.MaxEpoch: return True
        if epoch >= self.MinEpoch:
            params = self.GetAI()
            if params["AIrate"]<self.Threshold: return True
        return False
