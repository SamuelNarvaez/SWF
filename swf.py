import numpy as np
from trimesh import *
from utils import *
from constants import *

class SWF():
    def __init__(self,base,n=3,meshset=None):
        '''
        base : Trimesh 
            Trimesh sets the base mesh manually
        
        n : int
            finest mesh subdivision level
        meshset : iterable (optional)
            If, for example, you had a particular set of manually subdivided meshes that were compatible, they could be provided 
            here instead of generating the subdivisions automatically. It is important to note that n subdivisions will still occur. If this is not taken into account, it could result in a long runtime if you provide a relatively dense mesh in the meshset.
        
        '''
        self.base = base
        self.n = int(n)
        if meshset is not None:
            self.n += len(meshset)

        self.meshes = []
        if meshset is not None:
            self.meshes = meshset
            current = self.meshes[-1]
        else:
            current = self.base
        for i in range(int(n)):
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

    def encode(self,data,truncation_level=0):
        """
    Encode data in the fine representation to the coarse representation using the encoding filter at the given truncation level
    
    Parameters
    ----------
    data : (n,) float
      some signal defined over the finest level of mesh
    truncation level : int
      level at which to encode, default 0 
    Returns
    ----------
    encoded : (shape of vertices at truncation level)
    """
        encoded = self.phi2s[truncation_level] @ data
        return encoded
    
    def interpolate(self,loc,hop_size=1):
        '''
        for a given point (loc), returns the triangular interpolation accross the three vertices of the nearest triangle PQR on the mesh. 
        For a vertex P and a query point S, the interpolation weight for a vertex P is calculated as the area of the sub-triangle SQR divided by the total area of the triangle PQR. 
        Parameters
        ----------
        loc : (n,3) float
          one or many query points 
        hop_size (optional) : int
          if loc is an array with n>1, for example: a 1-second panning of 48000 samples, the hop size can reduce the number of calculations performed 
          at the cost of spatial resolution in time. A hop size of 10 for example would reduce from 48000 to 4800 calculations 
        Returns
        ----------
        encoded : (shape of vertices at truncation level)
        '''
        loc = loc.reshape((-1,3))
        triangles = self.meshes[-1].vertices[self.meshes[-1].faces]
        closest, dist, ind = self.meshes[-1].closest_point_naive(loc[::hop_size])
        closest = np.repeat(closest,hop_size)
        dist = np.repeat(dist,hop_size)
        ind = np.repeat(ind,hop_size)

        PQR = triangles[ind]
    
        AreaPQR = AreaTRI(PQR) #Area of PQR

        PQ = PQR[:,1] - PQR[:,0] #get the PQ vector of the triangle PQR
        PR = PQR[:,2] - PQR[:,0] #get the PR vector of the triangle PQR
        normals = np.cross(PQ,PR) #get the normal vector for the plane defined by the triangle PQR
        unitNormals = normals/np.linalg.norm(normals,axis=1).reshape(-1,1) #normal vector of unit length defined by PQR
        scalarDist = np.sum(unitNormals*(loc-PQR[:,0,:]),axis=1) #scalar distance from panning point to plane along the normal
        projection = loc - scalarDist.reshape(-1,1)*unitNormals #projection of panning point onto the plane defined by triangle PQR

        S = projection.reshape(-1,1,3) #reshaped for use in the area calculations

        SQR = np.hstack((S,PQR[:,1:,:])) #The triangle SQR defined by the panning point S and its two furthest neighbors
        PSR = np.hstack((PQR[:,0,:].reshape(-1,1,3),S,PQR[:,2,:].reshape(-1,1,3))) #The triangle PSR defined by S and its closest and furthest neighbors
        PQS = np.hstack((PQR[:,:2,:],S)) #The triangle PQS defined by S and its two closest neighbors

        AreaSQR = AreaTRI(SQR) #area of SQR
        AreaPSR = AreaTRI(PSR) #area of PSR
        AreaPQS = AreaTRI(PQS) #area of PQS

        interpolation = np.vstack((AreaSQR/AreaPQR,AreaPSR/AreaPQR,AreaPQS/AreaPQR)).T 
        interpolation = interpolation/interpolation.sum(axis=1).reshape(-1,1)
        fine = np.zeros((self.meshes[-1].vertices.shape[0],loc.shape[0]))
        fine[self.meshes[-1].faces[ind],np.arange(ind.shape[0]).reshape(-1,1)] = interpolation
        
        return fine
