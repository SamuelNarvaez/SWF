# Spherical Wavelet Format

About this Repo:

This is a working repository for ongoing research on Spherical Wavelet Format (SWF), a spatial audio format invented by my advisors Davide Scaini and Daniel Arteaga, as described in :

> D. Scaini and D. Arteaga, “Wavelet-Based Spatial Audio Format” J. Audio Eng. Soc., vol. 68, no. 9, pp. 613–627, (2020 September.). DOI: https://doi.org/10.17743/jaes.2020.0049

I am actively developing this project with the hopes of publishing a robust python library for playing around with SWF.

At the moment, everything works, but lacks good exception handling and I urge any user to navigate with caution. Feel free to reach out to me with any questions!

If you're a Max 8 user, the included max patch is a great place to start exploring this work. 

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

If you didn't exactly understand why we need everything in trimesh.py, that's okay. At some point it's just details. In swf.py we attempt to abstract away a useful amount of detail. swf.py builds SWF objects, which is a representation of a complete SWF format. In an SWF, we have:
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


