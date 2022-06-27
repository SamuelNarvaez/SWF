import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

def checkRelation(a,b,c):
    """
    Given three parameters: a,b,c check that they satisfy 2a + 2b + 4c == 1
    -----------
    a : float
    b : float
    c : float
    
    Returns
    -----------
    bool 
    """
    return np.isclose(2*a + 2*b + 4*c, 1)

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

def cost(SWF,wl,wt,level_to_optimize=0):
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
    encoder = SWF.phi2s[level_to_optimize]
    if level_to_optimize == 0:
        opt_level_vertices = SWF.base.vertices
    else:
        opt_level_vertices = SWF.meshes[level_to_optimize-1].vertices
    N = SWF.meshes[-1].vertices.shape[0]
    E = np.zeros([N,1])
    for i in range(N):
        onesource = np.zeros([N,1])
        onesource[i] = 1
        ui = SWF.meshes[-1].vertices[i]
        c0 = encoder@onesource
        V_ = np.sum(c0 * opt_level_vertices,axis=0)#velocity vector for each vertex
        Vl = np.sum(V_ * ui)
        Vt = np.linalg.norm(np.cross(V_,ui))
        Ei = wl*((Vl-1)**2) + wt*(Vt**2) #the cost as computed for a source at vertex i
        E[i]=Ei
    cost = np.sum(E)/N
    return cost

def check_sum_to_1(mat,axis):
    """
    Given a matrix, check that the sum is 1 along axis
    
    Parameters
    -----------
    mat : np.array
      matrix that we want to check sums to one along some axis
    axis : int
      axis along which to sum
    Returns
    -----------
    bool
    """
    return np.isclose(np.sum(mat,axis=axis),np.ones(mat.shape[(axis+1)%2]))

def toCartesian(point):
    """
    Given a point in spherical coordinates, convert to cartesian
    
    Parameters
    -----------
    point : np.array(r,a,e)
      radius, azimuth, elevation
    Returns
    -----------
    np.array(x,y,z)
    
    """
    x = point[0]*np.cos(point[1])*np.sin(point[2])
    y = point[0]*np.sin(point[1])*np.sin(point[2])
    z = point[0]*np.cos(point[2])
    return np.array([x,y,z])

def weights3D(mesh, weights, name=''):  
    """
    plot weights over the vertices of a mesh
    
    Parameters
    -----------
    mesh : Trimesh
      the mesh over which to plot
    weights : np.arr
      array of weights, same length as vertices in the mesh 
      
    Returns
    -----------
    plotly fig
    
    """
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
    
def PlotFilters(meshset):
    """
    plot first row/column of A,B,P,Q filters for every mesh in meshset
    
    Parameters
    -----------
    meshset : iterable of Trimesh
      the meshes from which to plot
      
    Returns
    -----------
    matplotlib fig
    
    """
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    for mesh in (meshset):
        plane = np.isclose(mesh.vertices[:,2],np.zeros(mesh.vertices[:,2].shape))
        P,Q,A,B = mesh.filters
        azimuth = np.arctan2(mesh.vertices[:,1],mesh.vertices[:,0])[plane].flatten()
        sorts = np.argsort(azimuth)
        axs[0,0].plot(azimuth[sorts],A[0,:][plane].flatten()[sorts],'--o',label=f'Level {mesh.level-1}')

        axs[0,1].plot(azimuth[sorts],B[0,:][plane].flatten()[sorts],'--o',label=f'Level {mesh.level-1}')

        axs[1,0].plot(azimuth[sorts],P[:,0][plane].flatten()[sorts],'--o',label=f'Level {mesh.level-1}')

        axs[1,1].plot(azimuth[sorts],Q[:,0][plane].flatten()[sorts],'--o',label=f'Level {mesh.level-1}')

    axs[0,0].set_title('First Row of A')
    axs[0,1].set_title('First Row of B')
    axs[1,0].set_title('First Column of P')
    axs[1,1].set_title('First Column of Q')
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    fig.tight_layout()

def PlotWavelets(SWF,idx=0):
      """
    plot wavelets and scaling functions for a given base vertex within a given SWF
    
    Parameters
    -----------
    SWF : SWF object
      the format of interest
    idx : int
      the index of the vertex to graph
      
    Returns
    -----------
    matplotlib fig
    
    """
    fig, axs = plt.subplots(2,2,figsize=(15,15))
    for i in range(0,SWF.n):
        mesh = SWF.meshes[-1]
        plane = np.isclose(mesh.vertices[:,2],np.zeros(mesh.vertices[:,2].shape))
        phi,psi,dualphi,dualpsi = (SWF.phis[i],SWF.psis[i],SWF.phi2s[i],SWF.psi2s[i]) 
        azimuth = np.arctan2(mesh.vertices[:,1],mesh.vertices[:,0])[plane].flatten()
        sorts = np.argsort(azimuth)
        axs[0,0].plot(azimuth[sorts],dualphi[idx,:][plane].flatten()[sorts],'-',label=f'$\overline{{\phi^{i+1}_{idx}}}$')

        axs[0,1].plot(azimuth[sorts],dualpsi[idx,:][plane].flatten()[sorts],'-',label=f'$\overline{{\psi^{i+1}_{idx}}}$')

        axs[1,0].plot(azimuth[sorts],phi[:,idx][plane].flatten()[sorts],'-',label=f'$\phi^{i+1}_{idx}$')

        axs[1,1].plot(azimuth[sorts],psi[:,idx][plane].flatten()[sorts],'-',label=f'$\psi^{i+1}_{idx}$')

    axs[0,0].set_title('horizontal section of dual scaling function, first row')
    axs[0,1].set_title('horizontal section of dual wavelet, first row')
    axs[1,0].set_title('horizontal section of scaling function, first column')
    axs[1,1].set_title('horizontal section of wavelet, first column')
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    fig.tight_layout()

def AreaTRI(TRI):
      """
    Calculates the area of a triangle supplied as a 1x3x3 array
    
    Parameters
    -----------
    TRI : (n, 3, 3) float
      n triangles of which to calculate the area
      
    Returns
    -----------
    area : (n, 1) float
    
    """
    TR = TRI[:,1] - TRI[:,0] #get the TR vector of the triangle TRI
    TI = TRI[:,2] - TRI[:,0] #get the TI vector of the triangle TRI
    return np.linalg.norm(np.cross(TR,TI),axis=1)/2 #Area of the triangle TRI
        
def closest_point_corresponding(triangles, points):
    """
    Return the closest point on the surface of each triangle for a
    list of corresponding points.
    
    Parameters
    ----------
    triangles : (n, 3, 3) float
      Triangle vertices in space
    points : (n, 3) float
      Points in space
    Returns
    ----------
    closest : (n, 3) float
      Point on each triangle closest to each point
    """
    tol = 1e-12
    # check input triangles and points
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)

    # store the location of the closest point
    result = np.zeros_like(points)
    # which points still need to be handled
    remain = np.ones(len(points), dtype=bool)

    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = [1.0, 1.0, 1.0]

    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = points - a
    # this is a faster equivalent of:
    # diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    # is the point at A
    is_a = np.logical_and(d1 < tol, d2 < tol)
    if any(is_a):
        result[is_a] = a[is_a]
        remain[is_a] = False

    # check if P in vertex region outside B
    bp = points - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # do the logic check
    is_b = (d3 > -tol) & (d4 <= d3) & remain
    if any(is_b):
        result[is_b] = b[is_b]
        remain[is_b] = False

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < tol) &
             (d1 > -tol) &
             (d3 < tol) & remain)
    if any(is_ab):
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False

    # check if P in vertex region outside C
    cp = points - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tol) & (d5 <= d6) & remain
    if any(is_c):
        result[is_c] = c[is_c]
        remain[is_c] = False

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tol) & (d2 > -tol) & (d6 < tol) & remain
    if any(is_ac):
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < tol) &
             ((d4 - d3) > - tol) &
             ((d5 - d6) > -tol) & remain)
    if any(is_bc):
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False

    # any remaining points must be inside face region
    if any(remain):
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

    return result