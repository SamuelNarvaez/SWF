import numpy as np
from trimesh import *
from utils import *

layout704=np.array([[1,0,np.pi/2],[1,np.pi/6,np.pi/2],[1,-np.pi/6,np.pi/2],[1,np.pi/2,np.pi/2],[1,-np.pi/2,np.pi/2],[1,3*np.pi/4,np.pi/2],[1,-3*np.pi/4,np.pi/2],[1,np.pi/4,np.pi/4],[1,-np.pi/4,np.pi/4],[1,3*np.pi/4,np.pi/4],[1,-3*np.pi/4,np.pi/4]])
vertices704 = np.apply_along_axis(lambda x: toCartesian(x),1,layout704)
faces704 = np.array([[6,4,10],[10,4,8],[8,4,2],[8,2,0],[8,7,0],[7,0,1],[7,1,3],[7,3,9],[9,3,5],[10,7,9],[10,8,7],[10,6,5],[10,9,5]])
vertices301 = np.array([[1,0,0],[-1/2,np.sqrt(3)/2,0],[-1/2,-np.sqrt(3)/2,0],[0,0,1]])
faces301 = np.array([[0,1,3],[1,2,3],[2,0,3]])
verticesOCT = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
facesOCT = np.array([[1,2,4],[1,3,4],[3,0,4],[0,2,4],[1,3,5],[3,0,5],[0,2,5],[2,1,5]])

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
        presets = {'OCT': 
                   Trimesh(verticesOCT,facesOCT,ALPHA=0.609151,BETA=-0.015081,GAMMA=-0.047035),
                   '7.0.4':
                   Trimesh(vertices704,faces704,ALPHA=0.546418,BETA=0.036781,GAMMA=-0.0416),
                   '3.0.1':
                   Trimesh(vertices301,faces301,ALPHA=0.599363,BETA=0.033933,GAMMA=-0.066648),
                   #'5.1.4':
                   #Trimesh(ALPHA=1/2,BETA=1/8,GAMMA=-1/16),
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

    def encode(self,data):
        encoded = self.phi2s[0] @ data
        return encoded