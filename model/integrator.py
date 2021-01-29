import numpy as np
import torch

class Integrator:
    def __init__(self, Nd, L, Np=1024):
        
        self.Nd = Nd
        self.L  = L
        self.Np = Np
        
    def points(self):
        X = np.empty((self.Np,0))
        for dInd in np.arange(self.Nd):
            X = np.append(X,self.L[dInd,0]+np.diff(self.L,axis=1)[dInd]*np.random.rand(self.Np,1),axis=1)
        X = torch.from_numpy(X).float().cuda()
        return X
    
    def integrate(self, func, reltol=1e-3, abstol=1e-6, maxiter=1e3):
        
        A = np.prod(np.diff(self.L,axis=1),axis=0)
        A = torch.from_numpy(A).cuda()
        S = torch.zeros(1).cuda()
        for iterNum in np.arange(maxiter):
            X = self.points()
            Y = func(X)
            Y = A*Y.mean(axis=0)
#             print('Y: ' + str(Y))
            dif = torch.abs(Y-S)
            rel = torch.abs(dif/S) if torch.abs(S) > 0 else torch.ones(1)
            S += 1/(iterNum+1)*(Y-S)
#             print(str(iterNum) + ': ' + str(S))
            
#             print(' abs: ' + str(dif))
#             print(' rel: ' + str(rel))
            if (dif <= abstol) or (rel <= reltol):
                break
        return S
            
        