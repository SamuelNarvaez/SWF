import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sparse
            
def faces_to_edges(faces, return_index=False):
    """
    Given a list of faces (n,3), return a list of edges (n*3,2)
    Parameters
    -----------
    faces : (n, 3) int
      Vertex indices representing faces
    Returns
    -----------
    edges : (n*3, 2) int
      Vertex indices representing edges
    """
    faces = np.asanyarray(faces)
    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    return edges

def get_second_neighbors(adj):
    """
    Given a an adjacency matrix, return a matrix of second neighbors
    Parameters
    -----------
    adj : (n, n) int
      A matrix where a_ij == 1 iff node i is incident to node j in the graph
    Returns
    -----------
    second : (n, n) int
      A matrix where a_ij == 1 iff node i is a second neighbor to node j
    """
    numPaths = adj @ adj #A^2 gives the number of walks of length 2 at a_ij between vertex i and vertex j
    pathExists = numPaths > 0 #a_ij == True if there exists at least one walk of length two between vertices i and j
    second = pathExists.astype(int)-adj-np.eye(adj.shape[0]) #exclude vertices that are first neighbors, and identity (a_ii)
    return second

def get_third_neighbors(adj):
    """
    Given a an adjacency matrix, return a matrix of third neighbors
    Parameters
    -----------
    adj : (n, n) int
      A matrix where a_ij == 1 iff node i is incident to node j in the graph
    Returns
    -----------
    third : (n, n) int
      A matrix where a_ij == 1 iff node i is a third neighbor to node j
    """
    second = get_second_neighbors(adj)
    numPaths = adj @ adj @ adj #A^3 gives the number of walks of length 3 at a_ij between vertex i and vertex j
    pathExists = numPaths > 0 #a_ij == True if there exists at least one walk of length three between vertex i and j
    third = pathExists.astype(int)-second-adj-np.eye(adj.shape[0]) #exclude vertices that are first and second neighbors, and identity (a_ii)
    return third

class Trimesh():
    def __init__(self,vertices=None,faces=None,filters=None,level=0):
        """
        vertices : (n, 3) float
           Array of vertex locations
        faces : (m, 3) int
          Indexes of vertices which make up triangular faces
        filters : 4-tuple of matrices (P,Q,A,B)
            xxxxxxxxxxxxxxxxxxxxxxxx
            xxxxxxxxxxxxxxxxxxxxxxxx
            xxxxxxxxxxxxxxxxxxxxxxxx
        level : int
          Level of subdivision of the mesh
        """
        self.level = level
        if vertices is not None:
            self.vertices = StandardScaler(with_std=False).fit_transform(vertices)
        if self.level==0:
            for i in range(len(self.vertices)):
                self.vertices[i] = self.vertices[i]/np.linalg.norm(self.vertices[i])
        self.faces = faces
        
        if filters is None:
            # This assures a consistent shape for the (P,Q,A,B) Tuple. At the coarsest level, only P is defined. 
            
            self.identity = np.identity(self.vertices.shape[0])
        
            
            self.filters = (self.identity,None,None,None)
            
        elif len(filters)==1:
            self.identity = filters[0]
            self.filters = (self.identity,None,None,None)
            
        else:
            #This lets us iteratively construct the filters for each level
            
            self.identity = None
            self.filters = filters

    def __repr__(self):
        return f"mesh level {self.level}" + "\nnum vertices: \n" + str(self.vertices.shape[0])

    def liftingScheme(self,P0,Q0,A0,B0,adj):
        m = Q0.shape[1] #details
        n = P0.shape[1] #coarse
        #S is mxn matrix coarse -> details
        #T is nxm matrix details -> coarse

        S = (1/6)*adj[-m:,:n]
        T = (3/14)*adj[:n,-m:] + (1/7)*get_second_neighbors(adj)[:n,-m:] + (1/14)*get_third_neighbors(adj)[:n,-m:]
        
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
        
        S_ = (1/6)*adj[-m:,:n]
        T_ = (3/14)*adj[:n,-m:] + (1/7)*get_second_neighbors(adj)[:n,-m:] + (1/14)*get_third_neighbors(adj)[:n,-m:]
        
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
        modified : wether or not to use the modified (True) lifting scheme or the classic (False).
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

        #turn new midpoints into unit vectors w.r.t. the origin
        if project_to_sphere:
            for i in range(len(mid)):
                mid[i] = mid[i]/np.linalg.norm(mid[i])

        new_vertices = np.vstack((self.vertices, mid))
        
        if self.identity is not None:
            #if we're at the coarsest level 0
            
            P = np.vstack((self.identity,np.zeros((mid.shape[0],self.identity.shape[1]))))
        
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

        return Trimesh(new_vertices, new_faces, new_filters, self.level + 1)

if __name__ == "__main__":
    pass
