import numpy as np
from scipy.special import jv
import torch

class PipeFlow:
    def __init__(self, Wo=10, Po=-1, Pn=None):
        
        self.Wo = Wo
        self.Po = Po
        self.Pn = Pn
        

    def velocity(self, x, y, z, t):
        r = np.sqrt(x**2+y**2)
        w = -1/4*self.Wo**2*self.Po*(1-r**2)
        if not self.Pn==None:
            for n in (np.arange(len(self.Pn))+1):
                zn = complex(0,1)**(3/2)*n**(1/2)
                w += np.real(complex(0,1)*self.Pn[n-1]/n*(1-jv(0,self.Wo*zn*r)/jv(0,self.Wo*zn))*np.exp(2*np.pi*complex(0,1)*n*t))
        w = w*(r**2<=1)
        u = np.zeros_like(w)
        v = np.zeros_like(w)
        
        return u,v,w
    
    def pressure(self, x, y, z, t):
        dpdz = self.Po
        if not self.Pn==None:
            for n in (np.arange(len(self.Pn))+1):
                dpdz += np.real(self.Pn[n-1]*np.exp(2*np.pi*complex(0,1)*n*t))
        p = dpdz*z
        
        return p
    
    def fields(self, X):
        x = X[:,0].cpu().numpy()
        y = X[:,1].cpu().numpy()
        z = X[:,2].cpu().numpy()
        t = X[:,3].cpu().numpy()
        N = X.size()[0]
        u,v,w = self.velocity(x,y,z,t)
        p = self.pressure(x,y,z,t)
#         print('u: ' + str(u.shape))
        F = np.concatenate((u.reshape((N,1)),v.reshape((N,1)),w.reshape((N,1)),p.reshape((N,1))),axis=1)
        F = torch.from_numpy(F).float().cuda()
        
        return F
        