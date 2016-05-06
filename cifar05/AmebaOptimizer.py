import os,sys,shutil
import numpy as np
import random
import copy

class AmabaOptimizer:
    def __init__(self):
        self.NextParam = []
        self.InitParam = []
        #self.Results = []
        #self.Expects = []
        self.ResExp   = [] # ResExp[][0]: Param, ResExp[][1]: True if result, False if expectation, ResExp[][2]: Score
        self.ParamSpace = []
        pass

    def SetParam(self,name,xset,initial=None):
        """
        name : parameter name. Returned in Param in NextParamSet()
        xset : list. Trial set. Must be continuous without flip
        initial : initial parameter. If None, selected randomly
        """
        if not initial:
            initial = random.choice(xset)

        idx = len(self.ParamSpace)
        self.ParamSpace.append([name,xset])
        self.InitParam.append(xset.index(initial))

        return

    def TranslateToDict(self,Params):
        res = {}
        for idx,p in enumerate(Params):
            #print self.ParamSpace[idx]
            #print self.ParamSpace[idx][1]
            #print p
            #print self.ParamSpace[idx][1][p]
            res[self.ParamSpace[idx][0]] = self.ParamSpace[idx][1][p]
        return res

    def GetMaxParamSet(self):
        """
        Select the best parameter in result values
        """
        ResList = filter(lambda x:x[1]==True,self.ResExp)
        Param = sorted(ResList,reverse=True,key=lambda x: x[2])[0]
        #print Param
        return self.TranslateToDict(Param[0]),Param[2]

    def GetNextParam(self):
        """
        Select the best parameter in expect values
        """
        if self.ResExp:
            ExpList = filter(lambda x:x[1]==False,self.ResExp)
            if len(ExpList)==0: return None
            Param = sorted(ExpList,reverse=True,key=lambda x: x[2])[0][0]
        else:
            Param = self.InitParam
        self.NextParam = Param
        print Param
        return self.TranslateToDict(Param)

    def SetResult(self,Result):
        param = self.NextParam
        if self.GetPosInList(param): del self.ResExp[self.GetPosInList(param)] # Delete if exist (as expectation)
        self.ResExp.append([param,True,Result])

        #print self.ResExp

        for idx,_ in enumerate(self.ParamSpace):
            param_p1 = self.ParamShift(param,idx,shift=+1)
            param_m1 = self.ParamShift(param,idx,shift=-1)
            #print param_p1,param_m1
            pos_p1   = self.GetPosInList(param_p1)
            pos_m1   = self.GetPosInList(param_m1)

            item_p1  = self.GetParamInList(pos_p1)
            item_m1  = self.GetParamInList(pos_m1)

            func = max

            #print param_p1,param_m1
            #print pos_p1,pos_m1
            #print item_p1,item_m1

            if   param_p1!=None and param_m1==None and item_p1==None and item_m1==None:
                self.ResExp.append([param_p1,False,Result])
            elif param_p1!=None and param_m1==None and item_p1!=None and item_m1==None and item_p1[1]==False:
                self.ResExp[pos_p1][2] = func(self.ResExp[pos_p1][2],Result)
            elif param_p1==None and param_m1!=None and item_p1==None and item_m1==None:
                self.ResExp.append([param_m1,False,Result])
            elif param_p1==None and param_m1!=None and item_p1==None and item_m1!=None and item_m1[1]==False:
                self.ResExp[pos_m1][2] = func(self.ResExp[pos_m1][2],Result)
            elif param_p1!=None and param_m1!=None and item_p1==None and item_m1==None:
                self.ResExp.append([param_p1,False,Result])
                self.ResExp.append([param_m1,False,Result])
            elif param_p1!=None and param_m1!=None and item_p1!=None and item_m1==None and item_p1[1]==False:
                self.ResExp[pos_p1][2] = func(self.ResExp[pos_p1][2],Result)
                self.ResExp.append([param_m1,False,Result])
            elif param_p1!=None and param_m1!=None and item_p1!=None and item_m1==None and item_p1[1]==True:
                self.ResExp.append([param_m1,False,Result-(self.ResExp[pos_p1][2]-Result)])
            elif param_p1!=None and param_m1!=None and item_p1==None and item_m1!=None and item_m1[1]==False:
                self.ResExp[pos_m1][2] = func(self.ResExp[pos_m1][2],Result)
                self.ResExp.append([param_p1,False,Result])
            elif param_p1!=None and param_m1!=None and item_p1==None and item_m1!=None and item_m1[1]==True:
                self.ResExp.append([param_p1,False,Result-(self.ResExp[pos_m1][2]-Result)])
            elif param_p1!=None and param_m1!=None and item_p1!=None and item_m1!=None and item_p1[1]==False and item_m1[1]==False:
                self.ResExp[pos_m1][2] = func(self.ResExp[pos_m1][2],Result)
                self.ResExp[pos_p1][2] = func(self.ResExp[pos_p1][2],Result)
            elif param_p1!=None and param_m1!=None and item_p1!=None and item_m1!=None and item_p1[1]==False and item_m1[1]==True:
                self.ResExp[pos_p1][2] = func(self.ResExp[pos_p1][2],Result-(self.ResExp[pos_m1][2]-Result))
            elif param_p1!=None and param_m1!=None and item_p1!=None and item_m1!=None and item_p1[1]==True and item_m1[1]==False:
                self.ResExp[pos_m1][2] = func(self.ResExp[pos_m1][2],Result-(self.ResExp[pos_p1][2]-Result))
            else:
                pass

        #print self.ResExp
        return

    def GetParamInList(self,pos):
        if pos==None:
            return None
        else:
            return self.ResExp[pos]

    def GetPosInList(self,param):
        """
        param : param to find
        """
        if param==None:
            return None

        for i, val in enumerate(self.ResExp):
            p,_,r = val
            if param == p:
                return i
        else:
            return None

    def ParamShift(self,param,idx,shift):
        assert (shift==+1 or shift==-1), "Shift should be +1 or -1"

        k = param[idx]

        if (shift<0 and k==0) or (shift>0 and k==(len(self.ParamSpace[idx][1])-1)):
            return None

        newparam = copy.deepcopy(param)
        newparam[idx] += shift

        return newparam


if __name__=="__main__":
    def ff(x1,x2):
        return 100.-(x1-0.)**2-(x2-3.)**2
    a = AmabaOptimizer()
    a.SetParam("x1",[1,2,3,4,5],initial=3)
    a.SetParam("x2",[1,2,3,4,5],initial=3)

    Num = []

    for i in range(100):
        Next = a.GetNextParam()
        if Next == None:
            print i
            break
        else:
            print Next
        res = ff(**Next)
        a.SetResult(res)
        Num.append(res)
        print a.GetMaxParamSet()
    print Num
