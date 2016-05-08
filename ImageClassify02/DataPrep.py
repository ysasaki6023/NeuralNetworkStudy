# vim:fileencoding=utf-8
import os,sys,shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import h5py


class DataPrep:
    def __init__(self):
        self.xUnit = 32
        self.yUnit = 32
        self.cUnit = 3
        self.MaxImage = 200000
        self.nCrops   = 3
        self.Data     = np.zeros((self.MaxImage*self.nCrops,self.xUnit*self.yUnit*self.cUnit),dtype=np.float32)
        self.Label    = np.zeros((self.MaxImage*self.nCrops),dtype=np.int32)
        self.ImIndex  = np.zeros((self.MaxImage*self.nCrops),dtype=np.int32)
        self.AgIndex  = np.zeros((self.MaxImage*self.nCrops),dtype=np.int32)
        self.dataURL  = []
        self.imageClass = []
        return
    def load(self):
        self.imageClass = []
        ImIndex = 0
        allIndex = 0
        # count files
        rcount = 0
        for root,dirs,files in os.walk("data"):
            print root
            #if rcount>=3: break
            rcount += 1
            for fname in files:
                # Load data
                img = Image.open(os.path.join(root,fname))
                if np.asarray(img).ndim != 3: continue # Not correct

                clsName = root.split("/")[1]
                if not (clsName in self.imageClass):
                    self.imageClass.append(clsName)
                label = self.imageClass.index(clsName)

                size_x = size_y = min(img.size[0],img.size[1])
                cent_x = int(img.size[0]/2)
                cent_y = int(img.size[1]/2)
                
                img1 = img.crop((cent_x-size_x/2,cent_y-size_y/2,cent_x+size_x/2,cent_y+size_y/2))
                img2 = img.crop((              0,              0,         size_x,         size_y))
                img3 = img.crop((img.size[0]-size_x,img.size[1]-size_y, img.size[0], img.size[1]))

                imgs = [img1,img2,img3]
                
                for AgIndex,im in enumerate(imgs):
                    im = im.resize((self.xUnit,self.yUnit),Image.BICUBIC)
                    ar = np.asarray(im)
                    ar = ar.transpose(1,2,0)
                    self.Data [allIndex]   = ar.reshape(self.xUnit*self.yUnit*self.cUnit)
                    self.Label[allIndex]   = label
                    self.ImIndex[allIndex] = ImIndex
                    self.AgIndex[allIndex] = AgIndex
                    allIndex += 1
                ImIndex += 1
                self.dataURL.append(os.path.join(root,fname).replace("data/",""))
        self.Data    = self.Data[:allIndex]
        self.Label   = self.Label[:allIndex]
        self.ImIndex = self.ImIndex[:allIndex]
        self.AgIndex = self.AgIndex[:allIndex]
        with h5py.File("input.dat","w") as h:
            h.create_group("Original")
            h.create_group("Info")
            h.create_dataset("Original/Data",data=self.Data   ,dtype=np.float32)
            h.create_dataset("Info/Label"   ,data=self.Label  ,dtype=np.int32)
            h.create_dataset("Info/ImIndex" ,data=self.ImIndex,dtype=np.int32)
            h.create_dataset("Info/AgIndex" ,data=self.AgIndex,dtype=np.int32)
            dt = h5py.special_dtype(vlen=unicode)
            h.create_dataset("Info/imageClassName"  ,data=self.imageClass,  dtype=dt)
            h.create_dataset("Info/imageClassIndex" ,data=range(len(self.imageClass)),dtype=np.int32)
            h.create_dataset("Info/imageURL"        ,data=self.dataURL,  dtype=dt)
            h.flush()
        return

if __name__=="__main__":
    dp = DataPrep()
    dp.load()
