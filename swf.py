import numpy as np
from trimesh import *

class SWF():
    def __init__(self,base='oct',n=3,l=0):
        '''
        base : string or Trimesh 
            string points to some predefined base mesh
            Trimesh sets the base mesh manually
        
        n : int
            finest mesh subdivision level
        l : int
            truncation level
        
        '''
        presets = {'oct': Trimesh(np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]),np.array([[1,2,4],[1,3,4],[3,0,4],[0,2,4],[1,3,5],[3,0,5],[0,2,5],[2,1,5]]),ALPHA=1/2,BETA=1/8,GAMMA=-1/16),
                   }
        if type(base) is str:
            self.base = presets[base]
        else: 
            self.base = base
        self.n = int(n)
        self.l = int(l)
        self.meshes = []
        current = self.base
        for i in range(self.n):
            result = current.subdivide()
            self.meshes.append(result)
            current = result
        self.Ps = [m.filters[0] for m in self.meshes]
        self.Qs = [m.filters[1] for m in self.meshes]
        self.As = [m.filters[2] for m in self.meshes]
        self.Bs = [m.filters[3] for m in self.meshes]
        self.phis = [self.phi(j) for j in range(self.n)]
        self.psis = [self.psi(j) for j in range(self.n)]
        self.phi2s = [self.phi2(j) for j in range(self.n)]
        self.psi2s = [self.psi2(j) for j in range(self.n)]
        
    def phi(self,j):
        '''
        computes the direct scaling function as (P_n*...*P_j+2*P_j+1)
        
        j : int [0,n-1] 
            level
        '''
        result = np.eye(self.Ps[j].shape[1])
        for P in self.Ps[j:]:
            result = P @ result
        return result
    def psi(self,j): 
        '''
        computes the direct wavelet as (P_n*...*P_j+2*Q_j+1)
        
        j : int [0,n-1] 
            level
        '''
        result = self.Qs[j]
        for P in self.Ps[j+1:]:
            result = P @ result
        return result
    def phi2(self,j):
        '''
        computes the dual scaling function as (A_j+1*A_j+2*...*A_n)
        
        j : int [0,n-1] 
            level
        '''
        result = np.eye(self.As[-1].shape[1])
        for A in self.As[j:][::-1]:
            result = A @ result
        return result
    def psi2(self,j):
        '''
        computes the dual wavelet as (B_j+1*A_j+2*...*A_n)
        
        j : int [0,n-1] 
            level
        '''
        result = np.eye(self.As[-1].shape[1])
        for A in self.As[j+1:][::-1]:
            result = A @ result
        result = self.Bs[j] @ result
        return result