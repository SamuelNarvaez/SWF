import numpy as np
from numpy import inf
import scipy.sparse as sparse
from utils import *

class Trimesh():
    def __init__(self,vertices=None,faces=None,filters=None,level=0,ALPHA=1/2,BETA=1/8,GAMMA=-1/16,LAMBDA=1/6):
        """
        vertices : (n, 3) float
           Array of vertex locations
        faces : (m, 3) int
          Indexes of vertices which make up triangular faces
        filters : 4-tuple of matrices (P,Q,A,B)
            where N = n+m
            P - N x n : takes coarse to fine 
            Q - N x m : takes details to fine
            A - n x N : takes fine to coarse 
            B - m x N : takes fine to details 
        level : int
          Level of subdivision of the mesh
        ALPHA : float
            multiplicative parameter for first neighbors, used in constructing T
        BETA : float
            multiplicative parameter for second neighbors, used in constructing T
        GAMMA : float
            multiplicative parameter for third neighbors, used in constructing T
        LAMBDA : float
            multiplicative parameter for first neighbors, used in constructing S
        """
        self.level = level
        if vertices is not None:
            self.vertices = vertices
        if self.level==0:
            self.vertices = self.vertices/np.linalg.norm(self.vertices,axis=1).reshape(-1,1)
        self.faces = faces
        
        if filters is None:
            # This assures a consistent shape for the (P,Q,A,B) Tuple. At the coarsest level P,Q,A, and B are undefined. 
            
            eye = np.identity(self.vertices.shape[0])
            
            self.filters = (eye,eye,eye,eye)
            
        else:
            self.filters = filters
        
        #set lifting parameters
        
        if checkCoeffRelations(ALPHA,BETA,GAMMA):
            self.ALPHA = ALPHA
            self.BETA = BETA
            self.GAMMA = GAMMA
            self.LAMBDA = LAMBDA
        else:
            raise ValueError('lifting coefficients do not satisfy the relation: 2a+2b+4c=1')  

    def __repr__(self):
        return f"mesh level {self.level}" + "\nnum vertices: \n" + str(self.vertices.shape[0])

    def liftingScheme(self,P0,Q0,A0,B0,adj):
        m = Q0.shape[1] #details
        n = P0.shape[1] #coarse
        #S is mxn matrix coarse -> details
        #T is nxm matrix details -> coarse
        adj2 = get_second_neighbors(adj)
        adj3, adj4 = get_third_neighbors(adj)
        
        np.seterr(divide='ignore', invalid='ignore') #the following computations regularize the parameters (Alpha,Beta,Gamma,Delta) for first, second, third and fourth neighbors for each of the details points, using the number of neighbors they actually have, i.e. the topology of the neighborhood of each point. 
        ALPHA = 2*self.ALPHA/np.sum(adj[-m:,:n],axis=1)
        BETA = 2*self.BETA/np.sum(adj2[-m:,:n],axis=1)
        GAMMA = 4*self.GAMMA/np.sum(adj3[-m:,:n],axis=1)
       
        ALPHA[np.isnan(ALPHA)] = 0
        ALPHA[ALPHA == -inf] = 0
        BETA[np.isnan(BETA)] = 0
        BETA[BETA == -inf] = 0
        GAMMA[np.isnan(GAMMA)] = 0
        GAMMA[GAMMA == -inf] = 0
        
        S = self.LAMBDA * adj[-m:,:n]
        T = ALPHA * adj[:n,-m:] + BETA * adj2[:n,-m:] + GAMMA * adj3[:n,-m:] + DELTA * adj4[:n,-m:]
        
        Im = np.identity(S.shape[0]) #mxm identity matrix
        In = np.identity(T.shape[0]) #nxn identity matrix
        
        P = P0 + Q0@S
        Q = -P0 @ T + Q0 @ (Im - S@T)
        A = (In - T@S)@A0 + T@B0
        B = B0-S@A0
        
        return P,Q,A,B

    def modliftingScheme(self,P0,Q0,A0,B0,adj):
        m = Q0.shape[1] #details
        n = P0.shape[1] #coarse
        #S_ is mxn matrix coarse -> details
        #T_ is nxm matrix details -> coarse
        adj2 = get_second_neighbors(adj)
        adj3 = get_third_neighbors(adj)
        
        np.seterr(divide='ignore', invalid='ignore') #the following computations regularize the parameters (Alpha,Beta,Gamma) for first, second and third neighbors for each of the details points, using the number of neighbors they actually have, i.e. the topology of the neighborhood of each point. 
        ALPHA = 2*self.ALPHA/np.sum(adj[-m:,:n],axis=1)
        BETA = 2*self.BETA/np.sum(adj2[-m:,:n],axis=1)
        GAMMA = 4*self.GAMMA/np.sum(adj3[-m:,:n],axis=1)
        
        #get rid of nans and infs if we have any.
        ALPHA[np.isnan(ALPHA)] = 0
        ALPHA[ALPHA == -inf] = 0
        BETA[np.isnan(BETA)] = 0
        BETA[BETA == -inf] = 0
        GAMMA[np.isnan(GAMMA)] = 0
        GAMMA[GAMMA == -inf] = 0
       
        S_ = self.LAMBDA * adj[-m:,:n]
        T_ = ALPHA * adj[:n,-m:] + BETA * adj2[:n,-m:] + GAMMA * adj3[:n,-m:]
        #print(np.all(check_sum_to_1(T_@B0,0)[n:]))
        
        Im = np.identity(S_.shape[0]) #mxm identity matrix
        In = np.identity(T_.shape[0]) #nxn identity matrix
        
        P = Q0 @ S_ + P0 @ (In - T_@S_)
        Q = Q0 - P0 @ T_
        A = A0 + T_ @ B0
        B = (Im - S_@T_)@B0 - S_@A0
        
        return P,Q,A,B

    def subdivide(self, project_to_sphere = True, modified = True):
        """
        Subdivide a mesh into smaller triangles,
        Carry out one iteration of lifting scheme on the 
        trivial interpolating filter associated with the subdivided mesh,
        return a new trimesh object
        
        Parameters
        ------------
        project_to_sphere :
            if True : new vertices generated by subdivision
                will be projected to the surface of the unit sphere
            if False : new vertices will stay inline with their two parent vertices.
        modified : wether or not to use the modified lifting scheme (True) or the unmodified lifting scheme (False).
          if True: calls modliftingScheme() to construct non-trivial P,Q,A,B
          if False: calls liftingScheme() to construct non-trivial P,Q,A,B
          
        Returns
        ----------
        New subdivided trimesh object
        """
        
        face_index = np.arange(len(self.faces))

        # find the unique edges of our faces subset
        edges = np.sort(faces_to_edges(self.faces), axis=1)
        _, unique, inverse, counts = np.unique(
            edges,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0)
        # then only produce one midpoint per unique edge
        mid = self.vertices[edges[unique]].mean(axis=1) #new vertices ordered by unique edges
        mid_idx = inverse.reshape((-1, 3)) + len(self.vertices) 
        
        # the new faces with correct winding
        f = np.column_stack([self.faces[:, 0],
                             mid_idx[:, 0],
                             mid_idx[:, 2],
                             mid_idx[:, 0],
                             self.faces[:, 1],
                             mid_idx[:, 1],
                             mid_idx[:, 2],
                             mid_idx[:, 1],
                             self.faces[:, 2],
                             mid_idx[:, 0],
                             mid_idx[:, 1],
                             mid_idx[:, 2]]).reshape((-1, 3))
        # add the 3 new faces per old face

        new_faces = np.vstack((self.faces, f[len(face_index):]))
        # replace the old face with a smaller face
        new_faces[face_index] = f[:len(face_index)]

        new_vertices = np.vstack((self.vertices, mid))
        
         #turn new midpoints into unit vectors w.r.t. the origin
        if project_to_sphere:
            new_vertices = new_vertices/np.linalg.norm(new_vertices,axis=1).reshape(-1,1)
        
        if self.level==0:
            eye = self.filters[0]
            
            P = np.vstack((eye,np.zeros((mid.shape[0],eye.shape[1]))))
        
            Q = np.vstack((np.zeros((P.shape[1],P.shape[0]-P.shape[1])),np.identity(P.shape[0]-P.shape[1])))
        
            A = P.T
        
            B = Q.T
            
        else:
            (P,Q,A,B) = self.filters
            
            PQ = np.hstack((P,Q)) #this is a throwaway matrix, but its dimension simplifies the following expression

            P = np.vstack((np.identity(PQ.shape[1]),np.zeros((mid.shape[0],PQ.shape[1]))))

            Q = np.vstack((np.zeros((P.shape[1],P.shape[0]-P.shape[1])),np.identity(P.shape[0]-P.shape[1])))
            
            A = P.T
        
            B = Q.T
            
        new_edges = np.sort(faces_to_edges(new_faces), axis=1)
        _1, unique1, inverse1, counts1 = np.unique(
            new_edges,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0)
        
        arr = new_edges[unique1]
        shape = (new_vertices.shape[0],new_vertices.shape[0])
        #create adjacency matrix including the newly generated vertices
        adj = sparse.coo_matrix((np.ones((arr.shape[0]*2)), (np.hstack((arr[:, 0],arr[:,1])), np.hstack((arr[:, 1],arr[:,0])))), 
                                shape=shape,dtype=arr.dtype).toarray() 
        if modified:
            P,Q,A,B = self.modliftingScheme(P,Q,A,B,adj)
            
        else:
            P,Q,A,B = self.liftingScheme(P,Q,A,B,adj)
        
        new_filters = (P,Q,A,B)

        return Trimesh(new_vertices, new_faces, new_filters, self.level + 1, ALPHA=self.ALPHA, BETA=self.BETA, GAMMA=self.GAMMA, LAMBDA=self.LAMBDA)

if __name__ == "__main__":
    pass
