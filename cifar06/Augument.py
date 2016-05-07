import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

class Augumentation:
    def __init__(self):
        #self.shiftRange  = [-2,+2]
        #self.skewRange   = [-0.01,+0.01]
        #self.scaleRange  = 0.2
        #self.skewRange   = 0.1
        #self.shiftRange  = 3 # 3pixel
        self.scaleRange  = 0.2
        self.skewRange   = 0.1
        self.shiftRange  = 3 # 3pixel
        self.Flip        = True
        pass
    def Aug(self,x):
        #return x
        """
        x : numpy image array
        """
        #plt.imshow(x[0].transpose(1,2,0))
        #plt.show()
        x = x.transpose(0,2,3,1)
        vmax = np.max(x)
        vmin = np.min(x)
        a = 255./(vmax-vmin)
        b = - a * vmin
        x = a*x+b
        r = np.zeros(x.shape,dtype=x.dtype)
        
        for i,img in enumerate(x):
            y = Image.fromarray(np.uint8(img),"RGB")
            #plt.imshow(y)
            #plt.show()
            a,e = [(1+random.uniform(-self.scaleRange,+self.scaleRange)) for k in range(2)]
            b,d = [random.uniform(-self.skewRange,+self.skewRange)*(a+e)/2 for k in range(2)]
            c,f = [random.uniform(-self.shiftRange,+self.shiftRange) for k in range(2)]
            y.transform( (y.size[0], y.size[1]), Image.AFFINE, (a,b,c,d,e,f), Image.BICUBIC)
            if self.Flip and random.random()>0.5:
                y = y.transpose(Image.FLIP_LEFT_RIGHT)
            r[i] = np.asarray(y)/255.
        r = r.transpose(0,3,1,2)
        r -= np.mean(r,axis=0)
        r /= np.std(r,axis=0)
        #print r[0]
        #plt.imshow(r[0].transpose(1,2,0))
        #plt.show()
        return r


