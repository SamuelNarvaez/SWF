import numpy as np
from sklearn.preprocessing import StandardScaler

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
    def __init__(self,vertices=None,faces=None,level=0):
        """
        vertices : (n, 3) float
           Array of vertex locations
        faces : (m, 3) int
          Indexes of vertices which make up triangular faces
        level : int
          Level of subdivision of the mesh
        """
        self.level = level
        if vertices is not None:
            self.vertices = StandardScaler(with_std=False).fit_transform(vertices)
        self.faces = faces
    def __repr__(self):
        return f"mesh level {self.level}" + "\nvertices: \n" + np.array_str(self.vertices) + "\nfaces: \n" + np.array_str(self.faces)

    def subdivide(self, project_to_sphere = False, face_index=None):
        """
        Subdivide a mesh into smaller triangles.
        Note that if `face_index` is passed, only those
        faces will be subdivided and their neighbors won't
        be modified making the mesh no longer "watertight."
        Parameters
        ------------
        face_index : faces to subdivide.
          if None: all faces of mesh will be subdivided
          if (n,) int array of indices: only specified faces
        Returns
        ----------
        new_vertices : (q, 3) float
          Vertices in space
        new_faces : (p, 3) int
          Remeshed faces
        """
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
            

        ###
        ### HERE THE LOOP WEIGHTING MASKS NEED TO BE APPLIED SO THAT THE COORDINATES FOR ALL VERTICES ARE RECOMPUTED
        ### THIS WILL LEAD OUR MESH TO GET SMOOTHER AND APPROACH C2 CONTINUITY OVER CONSECUTIVE SUBDIVISIONS
        ### FOR NOW, NEW VERTICES ARE LEFT IN THE SAME LINE AS THEIR SOURCES
        ###
        '''
                #for odd (new) vertices:
                for i in range(len(unique)):
                    if counts[i] == 1:
                        #then edge corresponds to crease or boundary, the geometric mean that was already computed will do.
                        continue
                    else:
                        #edge is an interior
                        interior = edges[unique[i]]
                        #find which two faces have interior edge
                        #find the third vertex of both faces
                        #compute the position of the new vertex as 3/8 the position of the endpoints of interior and 1/8 the two third vertices
                        #assign the computed position in the mid array

                #for even (old) vertices:
                    #if v is an interior:
                        #find all the adjecent faces
                        #weight all the unique points in the faces by beta, weight v's current position by 1-k*beta and sum
                    #if v is an exterior:
                        #weight v's two neighbors by 1/8 and v by 3/4 and sum


        '''



        new_vertices = np.vstack((self.vertices, mid))

        return Trimesh(new_vertices, new_faces, self.level + 1)

if __name__ == "__main__":
    mesh0 = Trimesh(np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]),np.array([[0,1,2],[1,3,2]]))
    print(mesh0)
    mesh1 = mesh0.subdivide()
