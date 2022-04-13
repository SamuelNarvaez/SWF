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
    second[second < 0] = 0 #crop negative values from the above operation
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
    pathExists = numPaths > 1 #a_ij == True if there exists at least two walks of length three between vertex i and j. Note that here we use at least two, because if there exists only one walk of length three between i and j, we consider this point a fourth neighbor. For more on this, see figure 3 of Interpolating Subdivision for Meshes with Arbitrary Topology by Zorin, Schroder, and Sweldens. 
    #fourthExists = numPaths == 1 #a_ij == True if there exists only one walk of length three between vertex i and j. This is key to 
    #fourth = fourthExists.astype(int)
    #fourth[fourth < 0] = 0
    third = pathExists.astype(int)-second-adj-np.eye(adj.shape[0]) #exclude vertices that are first and second neighbors, and identity (a_ii)
    third[third < 0] = 0
    
    return third

def cost(SWF,wp,wl,wt):
    '''
    Given a SWF defined over some mesh with some lifting coefficients, compute the acoustic pressure, longitudinal velocity, and 
    transverse velocity and compute a cost using these values.
    -----------
    SWF: SWF object
        the predefined SWF
    wp : int
        weight for the pressure component 
    wl : int
        weight for the longitudinal velocity component 
    wt : int
        weight for the transverse velocity component 
    Returns
    -----------
    cost : int
          cost value
    '''
    N = SWF.meshes[-1].vertices.shape[0]
    E = np.zeros([N,1])
    for i in range(N):
        onesource = np.zeros([N,1])
        onesource[i] = 1
        c0 = SWF.phi2s[0]@onesource
        V_ = c0 * SWF.base.vertices #velocity vector for each vertex
        Vt_ = np.sum(V_ * SWF.base.vertices,axis=1).reshape(-1,1)*SWF.base.vertices #transverse velocity component for each vertex
        Vl_ = V_ - Vt_ #logitudinal velocity component for each vertex
        Vt = np.linalg.norm(Vt_,axis=1) #norm of transverse velocity vector
        Vl = np.linalg.norm(Vl_,axis=1) #norm of longitudinal velocity vector
        Ei = wp*((np.sum(c0)-1)**2) + wl*((np.sum(Vl)-1)**2) + wt*(np.sum(Vt)**2) #the cost as computed for a source at vertex i
        E[i]=Ei
    cost = np.sum(E)/N
    return cost

def checkCoeffRelations(a,b,c):
    return 2*a + 2*b + 4*c == 1

def check_sum_to_1(mat,axis):
    return np.isclose(np.sum(mat,axis=axis),np.ones(mat.shape[(axis+1)%2]))

def toCartesian(point):
    x = point[0]*np.cos(point[1])*np.sin(point[2])
    y = point[0]*np.sin(point[1])*np.sin(point[2])
    z = point[0]*np.cos(point[2])
    return np.array([x,y,z])

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