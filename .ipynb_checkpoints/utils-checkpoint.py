from trimesh import *
import numpy as np 
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

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

def weights3D(mesh, weights, name=''):  
    x = mesh.vertices[:,0]
    y = mesh.vertices[:,1]
    z = mesh.vertices[:,2]
    i = mesh.faces[:,0]
    j = mesh.faces[:,1]
    k = mesh.faces[:,2]

    verts = mesh.vertices
    faces = mesh.faces

    fig = make_subplots(
              rows=1, cols=1, 
              subplot_titles=[f'{name}'],
              horizontal_spacing=0.02,
              specs=[[{"type": "scene"}]*1])  

    #plot surface triangulation
    tri_vertices = verts[faces]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_vertices:
        Xe += [T[k%3][0] for k in range(4)] + [ None]
        Ye += [T[k%3][1] for k in range(4)] + [ None]
        Ze += [T[k%3][2] for k in range(4)] + [ None]



    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, 
                            i=i, j=j, k=k, colorscale='matter_r' ,
                            colorbar_len=0.85,
                            colorbar_x=0.97,
                            colorbar_thickness=20,
                            intensity=weights,
                            text=weights,
                            intensitymode='vertex',
                            flatshading=True), 1, 1)
    lighting = dict(ambient=0.5,
                    diffuse=1,
                    fresnel=4,        
                    specular=0.5,
                    roughness=0.05,
                    facenormalsepsilon=0)
    lightposition=dict(x=100,
                       y=100,
                       z=10000)
    
    fig.data[0].update(lighting=lighting,
                       lightposition=lightposition)                         


    fig.update_layout(width=1000, height=600, font_size=10)
    fig.update_scenes(camera_eye_x=1.45, camera_eye_y=1.45, camera_eye_z=1.45);
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    fig.show()