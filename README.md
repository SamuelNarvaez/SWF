# Spherical Wavelet Format

About this Repo:

This is a working repository for ongoing research on Spherical Wavelet Format (SWF), a spatial audio format developed by my advisors Davide Scaini and Daniel Arteaga at Dolby Laboratories in Barcelona. In trimesh.py, you will find an implementation of a triangular mesh that, upon subdivision, builds the Encoding and Decoding filters P,Q,A,B as described in :

> D. Scaini and D. Arteaga, “Wavelet-Based Spatial Audio Format” J. Audio Eng. Soc., vol. 68, no. 9, pp. 613–627, (2020 September.). DOI: https://doi.org/10.17743/jaes.2020.0049

And supports updating the trivial interpolating filter via the Lifting Scheme or the Modified Lifting Scheme. More details to come in a publication soon.

In swf.py, the SWF class handles the creation of the wavelets and scaling functions and the dual wavelets and dual scaling functions as matrices for use in encoding/decoding as detailed in the paper. The goal is to use these classes to generate trivial decodings of spatial audio for reproduction from virtual sources with a long-term goal of being able to record directly in the format. 

The notebooks deal with various vizualization strategies to see the mesh and the filters, scaling functions, and wavelets as weights on the mesh––as well as a space for testing during development. 
