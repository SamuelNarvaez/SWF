import numpy as np
from trimesh import *
from swf import *
from utils import *
from scipy.optimize import minimize

class OptimalSWF():
    def __init__(self, vertices, faces, n=3, level_to_optimize=0):
        self.vertices = vertices
        self.faces = faces
        self.n = n
        self.level_to_optimize = level_to_optimize
        initial_guess = np.array([0.5,0])
        res = minimize(self.f,initial_guess)
        a,b = res.x
        c = (1-2*(a+b))/4
        self.model = SWF(Trimesh(self.vertices,self.faces,ALPHA=a,BETA=b,GAMMA=c), n=self.n)
        
    def f(self, coeffs):
        ALPHA, BETA = coeffs
        GAMMA = (1-2*(ALPHA+BETA))/4
        mesh = Trimesh(self.vertices,self.faces,ALPHA=ALPHA,BETA=BETA,GAMMA=GAMMA)
        model = SWF(mesh, n=self.n)
        return cost(model,1,1,self.level_to_optimize)
