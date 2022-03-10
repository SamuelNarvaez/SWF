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

def check_sum_to_1():
    pass