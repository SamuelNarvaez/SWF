import numpy as np

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
    def __init__(vertices=None,faces=None,level=0):
    """
    vertices : (n, 3) float
       Array of vertex locations
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    level : int
      Level of subdivision of the mesh
    """
    if vertices is not None:
            # (n, 3) float, set of vertices
            self.vertices = vertices
    if faces is not None:
            # (m, 3) int of triangle faces, references self.vertices
            self.faces =





'''
def subdivide(vertices,
              faces,
              face_index=None):
    """
    Subdivide a mesh into smaller triangles.
    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."
    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
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
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_index]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    _, unique, inverse = np.unique(
        edges,
        return_index=True,
        return_inverse=True)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

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
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    return new_vertices, new_faces

'''
