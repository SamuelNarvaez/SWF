# Spherical Wavelet Format

About this Repo:

This is a working repository for ongoing research on Spherical Wavelet Format (SWF), a spatial audio format invented by my advisors Davide Scaini and Daniel Arteaga, as described in :

> D. Scaini and D. Arteaga, “Wavelet-Based Spatial Audio Format” J. Audio Eng. Soc., vol. 68, no. 9, pp. 613–627, (2020 September.). DOI: https://doi.org/10.17743/jaes.2020.0049

I am actively developing this project with the hopes of publishing a robust python library for playing around with SWF. In the dev branch, you will find a folder called testing which has a lot of good example notebooks. 

At the moment, everything works, but lacks good exception handling and I urge any user to navigate with caution. Feel free to reach out to me with any questions!

If you're a Max 8 user, the included max patch is a great place to start exploring this work. 

In SWF, we decompose the soundfield by a set of basis functions called spherical wavelets. This is all handled by the library, and it is not neccessary to understand the mathematical details in order to generate and use a SWF format. I would, recommend you read the above reference if you're interested in learning more.

A particular Spherical Wavelet Format is defined by:
* A sequence of Subdivision Meshes such that every vertex of a mesh is a subset of the next mesh in the sequence.
* Encoding and Decoding filters P,Q, and A,B respectively, for each mesh in the sequence that satisfy biorthogonality relations, the set of all filters define a wavelet space.
* A truncation level, which specifies the order of wavelet decomposition


To use this library, first define a base triangular mesh that closely resembles your indtended speaker layout for reproduction. This can be done by passing the coordinates of the vertices in R3 as a numpy array and the faces as a numpy array of indices of vertices in the vertex array to the Trimesh constructor. Consider this example, with an octahedron as the base:

```
from trimesh import * 
import numpy as np

vertices_octahedron = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
faces_octahedron = np.array([[1,2,4],[1,3,4],[3,0,4],[0,2,4],[1,3,5],[3,0,5],[0,2,5],[2,1,5]])

base = Trimesh(vertices704,faces704)
```

Next, decide how many levels of subdivision (resulting in a higher spatial-resolution mesh) you would like to carry out –– 3 is a good place to start. You can subdivide by hand, by calling the subdivide method of the Trimesh object, which returns a new, subdivided Trimesh Object:

```
subdivided_once = base.subdivide() #Trimesh object
subdivided_twice = subdivided_once.subdivide() #Trimesh object
subdivided_threetimes = subdivided_twice.subdivide() #Trimesh object
```

This interface can be useful if you're interested in the details/having more control. Each call to subdivide is also generating the filters P,Q,A,B as described in the original paper via the lifting scheme. 

However, it is most likely you will want to use a higher-level interface to automate this proccess. The SWF class found in swf.py abstracts the subdivision procces and allows the specification of the number of subdivisions as an argument:

```
from swf import *

swf_3_subdivisions = SWF(Trimesh(vertices_octahedron,faces_octahedron), n=3) #SWF object
```

the SWF object holds the entire sequence of subdivison meshes, as well as the sequence of P,Q,A,B filters to transition data between them. For convenience, the wavelets, dual wavelets, scaling functions, dual scaling functions have been precomputed for each level and are stored in a list as an attribute. 

Similarly, if you want your SWF format to be optimized for some psychoacoustical properties, you can use the OptimalSWF class found in optimal.py to automatically subdivide and generate optimized filters along the way:

```
from optimal import *

optimized_3_subdivisions = OptimalSWF(vertices_octahedron,faces_octahedron,n=3).model #SWF object
```

The most common operations needed for a given swf are encoding, and interpolating. If we have a virtual source, and we want to encode it in SWF at location x,y,z:

```
loc = np.array([x,y,z])
interpolation = swf.interpolate(loc)
```
the interpolate method performs VBAP-style interpolation at the finest level of spatial resolution, and returns a vector defined over the finest level of mesh which can be multiplied against the signal of our virtual source to place it in space. To encode to our coarsest level of mesh:

```
fine = signal * interpolation
coarse = swf.encode(fine)
```
If our coarsest level of mesh is the same as our speaker array, we can send the resulting channels directly to the speakers. If we have encoded to some higher truncation level, or our mesh is not identical to the speaker array, some additional decoding step must be implemented and calculated at this point.

Here's a brief overview of the structure of the libary:

# trimesh.py
in trimesh.py, you'll find a data structure for a triangular mesh, based heavily on Mike Dawson-Haggerty's trimesh.

> @software{trimesh,
	author = {{Dawson-Haggerty et al.}},
	title = {trimesh},
	url = {https://trimsh.org/},
	version = {3.2.0},
	date = {2019-12-8},
}

The important method of the Trimesh object is subdivison. After specifying a base mesh, we can subdivide it automatically using the loop subdivision scheme, or the user can manually specify the next subdivision mesh. Subdivsion allows us to approximate the sphere (or hemisphere) with greater resolution–resulting in higher spatial resolution. Upon subdivision, a trimesh builds the Decoding and Encoding filters P,Q and A,B which allow us to represent data defined over the mesh at different levels of detail. 

More specifically, if we have data f defined over the fine (subdivided) level of mesh, applying the A filter to the data f gives us a spatially low-passed representation of it, c, defined over the coarse (pre-subdivision) mesh. Applying the B filter to the data f gives us a representation of all the details, d, not included in the spatially low-passed representation, its dimensionality being the number of points added by the subdivision scheme. 

Conversely, the P filter takes the coarse representation, c, and upsamples it over the fine mesh. Similarly, the Q filter takes the details d and upsamples it over the fine mesh. 

Here, to generate P,Q,A and B, we start with a trivial set of filters and then apply the Lifting Scheme, or in the case of this libary, a Modified Lifting Scheme better suited for spatial audio. The lifting scheme ensures that the generated filters are biorthogonal, meaning essentially that:

* There is no overlap in the vertices of the mesh that the non-corresponding filters act upon
* There is no vertex acted upon by the decoders that the corresponding encoder does not act upon
* There is no information that is left out of the encoding by A and B, and no information that cannot be decoded by P and Q.

# swf.py

If you didn't exactly understand why we need everything in trimesh.py, that's okay. At some point it's just details. In swf.py we attempt to abstract away a useful amount of detail. swf.py builds SWF objects, which is a representation of a complete SWF format. In an SWF object, we have:

* A sequence of Subdivision Meshes such that every vertex of a mesh is a subset of the next mesh in the sequence.
* Encoding and Decoding filters P,Q, and A,B respectively, for each mesh in the sequence that satisfy biorthogonality relations, the set of all filters define a wavelet space.
* A truncation level, which specifies the order of wavelet decomposition, for audio purpouses you can think of this as the point to which we decode to the speaker layout.

there are two mandatory arguments: base and level. base expects a Trimesh object, and the level defines how many iterations of subdivision to perform. Note that the time complexity of the subdivision increases exponentially, without much gain in spatial resolution after 3 iterations or so, so try to keep level less than or equal to 4 unless you want to wait a while. 

I recommend using a base mesh with vertices as close to your speaker layout as possible, that way you can use the trivial decoding from the base mesh and send the gains directly to your speakers. 

Note: If you plan on manually subdividing your base mesh, use the meshest argument when instantiating an SWF to provide all the manual subdivisions in an ordered list. You might want to manually subdivide to impute virtual points to correct issues with L/R symmetry in the triangulation of your base mesh, for example. 

# optimal.py

extends SWF, performs an optimization on the filter A for psychoacoustical properties. If you're not interested in all the details, I would start here. Generate an optimal SWF with a base mesh identical to your speaker layout. 

# OSCserver.py 

For a virtual source at location recieved over OSC, calculate a VBAP-style trilinear interpolation over the finest level of mesh and send the result over OSC. The interpolation must be encoded to the coarse mesh at the destination. Central to the functioning of the included Max Patch :)

# utils.py and constants.py

utility functions and constants used by the other classes 


