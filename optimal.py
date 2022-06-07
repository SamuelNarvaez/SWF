import numpy as np
from trimesh import *
from swf import *
from utils import *
from scipy.optimize import minimize

class OptimalSWF():
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        initial_guess = np.array([0.5,0])
        res = minimize(self.f,initial_guess)
        a,b = res.x
        c = (1-2*(a+b))/4
        self.model = SWF(Trimesh(self.vertices,self.faces,ALPHA=a,BETA=b,GAMMA=c), n=3)
        
    def f(self, coeffs):
        ALPHA, BETA = coeffs
        GAMMA = (1-2*(ALPHA+BETA))/4
        mesh = Trimesh(self.vertices,self.faces,ALPHA=ALPHA,BETA=BETA,GAMMA=GAMMA)
        model = SWF(mesh, n=3)
        return cost(model,1,1)
        
    def total_acoustic_pressure(self, virtual_source_loc):
        pass
    def energy(self, virtual_source_loc):
        pass
    def velocity(self, virtual_source_loc):
        pass
    def intensity(self, virtual_source_loc):
        pass