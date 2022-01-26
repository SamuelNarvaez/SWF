import numpy as np
from sklearn.preprocessing import StandardScaler

class PrintArray:

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __repr__(self):
        rpr = ('PrintArray(' +
               ', '.join([f'{name}={value}' for name, value in self._kwargs.items()]) +
               ')')
        return rpr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc != np.floor_divide:
            return NotImplemented
        a = inputs[0]
        with np.printoptions(**self._kwargs):
            print(a)
            
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

class Trimesh():
    def __init__(self,vertices=None,faces=None,weights=None,filters=None,level=0):
        """
        vertices : (n, 3) float
           Array of vertex locations
        faces : (m, 3) int
          Indexes of vertices which make up triangular faces
        weights : (n,1) float
          weights on each vertex
        filters : (n,n) float
            square matrix representing the subdivision filter for the weights
            if no initial filter is provided, identity is used.
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
        
        if weights is None:
            self.weights = np.ones((self.vertices.shape[0],1))
        else:
            self.weights = np.asanyarray(weights).reshape((-1,1))
        
        if filters is None:
            # This assures a consistent shape for the (P,Q,A,B) Tuple. At the coarsest level, only P is defined. 
            
            self.identity = np.identity(self.vertices.shape[0])
        
            
            self.filters = (self.identity,)
            
        elif len(filters)==1:
            self.identity = filters[0]
            self.filters = (self.identity,)
            
        else:
            #This lets us iteratively construct the filters for each level
            
            self.identity = None
            self.filters = filters

    def __repr__(self):
        return f"mesh level {self.level}" + "\nnum vertices: \n" + str(self.vertices.shape[0]) + "\nnum faces: \n" + str(self.faces.shape[0]) + "\nweights: \n" + np.array_str(self.weights) 

    def liftingScheme(self,P0,Q0,A0,B0):
        
        S = np.random.rand(Q0.shape[1],P0.shape[1]) #mxn matrix
        T = np.random.rand(P0.shape[1],Q0.shape[1]) #nxm matrix
        
        Im = np.identity(S.shape[0]) #mxm identity matrix
        In = np.identity(T.shape[0]) #nxn identity matrix
        
        P = P0 + Q0@S
        Q = -P0 @ T + Q0 @ (Im - S@T)
        A = (In - T@S)@A0 + T@B0
        B = B0-S@A0
        
        return P,Q,A,B

    def modliftingScheme(self,P0,Q0,A0,B0):
        
        S_ = np.random.rand(Q0.shape[1],P0.shape[1]) #mxn matrix
        T_ = np.random.rand(P0.shape[1],Q0.shape[1]) #nxm matrix
        
        Im = np.identity(S_.shape[0]) #mxm identity matrix
        In = np.identity(T_.shape[0]) #nxn identity matrix
        
        P = Q0 @ S_ + P0 @ (In - T_@S_)
        Q = Q0 - P0 @ T_
        A = A0 + T_ @ B0
        B = (Im - S_@T_)@B0 - S_@A0
        
        return P,Q,A,B

    def subdivide(self, project_to_sphere = False, modified = False, face_index=None):
        """
        Subdivide a mesh into smaller triangles.
        Note that if `face_index` is passed, only those
        faces will be subdivided and their neighbors won't
        be modified making the mesh no longer "watertight."
        Parameters
        ------------
        project_to_sphere :
            if True : new vertices generated by subdivision
                will be projected to the surface of the unit sphere
            if False : new vertices will stay inline with their two parent vertices.
        face_index : faces to subdivide.
          if None: all faces of mesh will be subdivided
          if (n,) int array of indices: only specified faces
        Returns
        ----------
        New subdivided trimesh object
        """
        pa = PrintArray(precision=2, linewidth=150, suppress=True)
        
        if face_index is None:
            face_index = np.arange(len(self.faces))
        else:
            face_index = np.asanyarray(face_index)

        # the (c, 3) int array of vertex indices
        faces_subset = self.faces[face_index]

        # find the unique edges of our faces subset
        edges = np.sort(faces_to_edges(faces_subset), axis=1)
        _, unique, inverse, counts = np.unique(
            edges,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0)
        # then only produce one midpoint per unique edge
        mid = self.vertices[edges[unique]].mean(axis=1)
        mid_idx = inverse.reshape((-1, 3)) + len(self.vertices)

        # the new faces_subset with correct winding
        f = np.column_stack([faces_subset[:, 0],
                             mid_idx[:, 0],
                             mid_idx[:, 2],
                             mid_idx[:, 0],
                             faces_subset[:, 1],
                             mid_idx[:, 1],
                             mid_idx[:, 2],
                             mid_idx[:, 1],
                             faces_subset[:, 2],
                             mid_idx[:, 0],
                             mid_idx[:, 1],
                             mid_idx[:, 2]]).reshape((-1, 3))
        # add the 3 new faces_subset per old face
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
            
        if modified:
            P,Q,A,B = self.modliftingScheme(P,Q,A,B)
            
        else:
            P,Q,A,B = self.liftingScheme(P,Q,A,B)
        
        new_weights = P @ self.weights
        
        new_filters = (P,Q,A,B)

        return Trimesh(new_vertices, new_faces, new_weights, new_filters, self.level + 1)

if __name__ == "__main__":
    mesh0 = Trimesh(np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]),np.array([[0,1,2],[1,3,2]]))
    print(mesh0)
    mesh1 = mesh0.subdivide()
