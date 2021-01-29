import numpy as np
import torch

class Sampler:
    def __init__(self, Np=1024):
        self.Np      = Np
        
    def points(self):
        X = np.random.rand(self.Np,1)
        X = torch.from_numpy(X).float().cuda()
        return X
        
class CylinderSampler(Sampler):
    def __init__(self, R, L, T, Np=1024):
        super().__init__()
        self.R  = R
        self.L  = L
        self.T  = T
        self.Np = Np
        
    def interior(self, Np=None):
        N = self.Np if Np==None else Np
        r  = np.random.rand(N,1)*self.R
        th = np.random.rand(N,1)*2*np.pi
        x  = r*np.cos(th)
        y  = r*np.sin(th)
        z  = np.random.rand(N,1)*self.L
        t  = np.random.rand(N,1)*self.T
        
        X = np.concatenate((x,y,z,t),axis=1)
        X = torch.from_numpy(X).float().cuda()
        
        return X
        
    def wall(self, Np=None):
        N = self.Np if Np==None else Np
        th = np.random.rand(N,1)*2*np.pi
        r  = np.ones_like(th)*self.R
        x  = r*np.cos(th)
        y  = r*np.sin(th)
        z  = np.random.rand(N,1)*self.L
        t  = np.random.rand(N,1)*self.T
        
        X = np.concatenate((x,y,z,t),axis=1)
        X = torch.from_numpy(X).float().cuda()
        
        return X
        
    def inlet(self, Np=None):
        N = self.Np if Np==None else Np
        r  = np.random.rand(N,1)*self.R
        th = np.random.rand(N,1)*2*np.pi
        x  = r*np.cos(th)
        y  = r*np.sin(th)
        z  = np.zeros_like(x)
        t  = np.random.rand(N,1)*self.T
        
        X = np.concatenate((x,y,z,t),axis=1)
        X = torch.from_numpy(X).float().cuda()
        
        return X
        
    def outlet(self, Np=None):
        N = self.Np if Np==None else Np
        r  = np.random.rand(N,1)*self.R
        th = np.random.rand(N,1)*2*np.pi
        x  = r*np.cos(th)
        y  = r*np.sin(th)
        z  = np.ones_like(x)*self.L
        t  = np.random.rand(N,1)*self.T
        
        X = np.concatenate((x,y,z,t),axis=1)
        X = torch.from_numpy(X).float().cuda()
        
        return X
        
    def initial(self, Np=None):
        N = self.Np if Np==None else Np
        r  = np.random.rand(N,1)*self.R
        th = np.random.rand(N,1)*2*np.pi
        x  = r*np.cos(th)
        y  = r*np.sin(th)
        z  = np.random.rand(N,1)*self.L
        t  = np.zeros_like(x)
        
        X = np.concatenate((x,y,z,t),axis=1)
        X = torch.from_numpy(X).float().cuda()
        
        return X
    
    def convertToTimeZero(self, X):
        X[:,3] = 0
        
        return X
    
    def wallNormal(self, X):
        np = X.size()[0]
        x = X[:,0]
        y = X[:,1]
        r = torch.sqrt(x**2+y**2)
        N = torch.zeros((np,3)).float().cuda()
        N[:,0] = y/r
        N[:,1] = x/r
        
        return N
    
    def outletNormal(self, X):
        np = X.size()[0]
        N = torch.zeros((np,3)).float().cuda()
        N[:,2] = 1
        
        return N
    
        
        