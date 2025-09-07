import math
from typing import Tuple

import numpy as np
import cv2

from .. import lerp, normalize_range


def smoothfunc(t):
	return t * t * t * ((6 * t - 15) * t + 10)

def raw_perlin_noise_2d(shape:Tuple[int,int], density:float, aspect:float=1., tileable:bool=False, normalized:bool=True) -> np.ndarray:
    """
        Generate a 2D numpy image of Perlin noise.

        > Note: Use `perlin_noise_2d` instead, to generate at lower resolution and upscale to target size.

        Parameters
        ----------
        shape : Tuple[int,int]
            The shape of the generated image.
        density : float
            Number of cells inside the image.
        aspect : float
            How many times is noise expanded along the image width. Default: 1.0
        tileable : bool
            Make the noise tileable along both axes. Default: False
        normalized : bool
            Normalize the values to the [-1,1] interval. Default: True

        Notes
        ----------
        Inspired by https://rtouti.github.io/graphics/perlin-noise-algorithm, following structure of https://en.wikipedia.org/wiki/Perlin_noise, optimized to perform faster and be less biased.
    """
    assert len(shape)==2, f'Please provide only two dimensions for the shape. Got: {shape}'

    # Resolve density pixels from value.
    density_shape = int(math.ceil(density))
    density_shape = np.array([density_shape, int(math.ceil(density_shape*aspect))], dtype=np.int32)

    # Define pixels grid.
    pixels = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1])), axis=2, dtype=np.float32)
    pixels = pixels.reshape(-1,2)

    # Map to the gradient field coordinates.
    grid = pixels / (np.array(shape)-1) * (density_shape-1)  # [0,density_shape-1]

    # Find corner locations in the gradient field.
    low = np.floor(grid).astype(np.int32)
    corner00 = low
    corner01 = low + np.array([[0, 1]], dtype=np.int32)
    corner10 = low + np.array([[1, 0]], dtype=np.int32)
    corner11 = low + 1

    # Calculate displacement from corners in gradiend field.
    displace00 = grid - corner00
    displace01 = grid - corner01
    displace10 = grid - corner10
    displace11 = grid - corner11

    # Construct gradients field.
    gradients = np.random.random((density_shape[0]+1, density_shape[1]+1, 2)) * 2 - 1
    if tileable:
        gradients[:,-1] = gradients[:,0]
        gradients[-1,:] = gradients[0,:]
        gradients[:,-2] = gradients[:,0]
        gradients[-2,:] = gradients[0,:]

    # Get gradients from gradient field.
    grad00 = gradients[corner00[:,0], corner00[:,1]]
    grad01 = gradients[corner01[:,0], corner01[:,1]]
    grad10 = gradients[corner10[:,0], corner10[:,1]]
    grad11 = gradients[corner11[:,0], corner11[:,1]]

    # Calculate dot between gradients and displacements.
    dot00 = (displace00 * grad00).sum(axis=1)
    dot01 = (displace01 * grad01).sum(axis=1)
    dot10 = (displace10 * grad10).sum(axis=1)
    dot11 = (displace11 * grad11).sum(axis=1)

    # Interpolate between dots using function with 1st derivative (maybe also 2nd derivative) equal to 0 close to grid points.
    uv = smoothfunc(grid - low)
    values = lerp(
        uv[:,0],
        lerp(uv[:,1], dot00, dot01),
        lerp(uv[:,1], dot10, dot11)
    )

    if normalized:
        values = normalize_range(values) * 2 - 1

    return values.reshape(shape)


def perlin_noise_2d(resolution_factor:float, shape:Tuple[int,int], density:float, aspect:float=1., tileable:bool=False, normalized:bool=True) -> np.ndarray:
    """
        Generate a 2D numpy image of Perlin noise at a lower resolution (for a fraction of original compute time) and rescale cubically to target size.
        The pattern is of equivalent density to original, however at larger factors some interpolation artifacts might appear.
        Prefer this method to original when dealing with large resolutions.

        Parameters
        ----------
        resolution_factor : float
            The factor of reduced resolution where the Perlin noise generation will take place. If `1`, directly calls the original mehtod.
        shape : Tuple[int,int]
            The shape of the generated image.
        density : float
            Number of cells inside the image.
        aspect : float
            How many times is noise expanded along the image width. Default: 1.0
        tileable : bool
            Make the noise tileable along both axes. Default: False
        normalized : bool
            Normalize the values to the [-1,1] interval. Default: True

        Notes
        ----------
        Inspired by https://rtouti.github.io/graphics/perlin-noise-algorithm, following structure of https://en.wikipedia.org/wiki/Perlin_noise, optimized to perform faster and be less biased.
    """
    if resolution_factor == 1.0:
         return perlin_noise_2d(shape, density=density, aspect=aspect, tileable=tileable, normalized=normalized)
    noise = perlin_noise_2d((int(shape[0]/resolution_factor), int(shape[1]/resolution_factor)), density=density, aspect=aspect, tileable=tileable, normalized=normalized)
    return cv2.resize(noise, shape, interpolation=2)  # Cubic interpolation for preserving curves.


def fractal_noise_2d(resolution_factor:float, shape:Tuple[int,int], density:float, aspect:float=1., tileable:bool=False, normalized:bool=True, octaves:int=1, persistence:float=0.5, lacunarity:float=2.0):
    """
        Generate a 2D numpy array of fractal (multi-octave) Perlin noise.

        Parameters
        ----------
        shape : Tuple[int,int]
            The shape of the generated array (tuple of two ints). This must be a multiple of lacunarity**(octaves-1)*density.
        density : Tuple[int,int]
            The number of periods of noise to generate along each axis (tuple of two ints). Note shape must be a multiple of (lacunarity**(octaves-1)*density).
        octaves : int
            The number of octaves in the noise. Default: 1.
        persistence : float
            The scaling factor between two octaves. Default: 0.5
        lacunarity : float
            The frequency factor between two octaves. Default: 2.0
        tileable : Tuple[bool,bool]
            If the noise should be tileable along each axis (tuple of two bools). Defaults to False.
    """
    noise = np.zeros(shape, dtype=np.float32)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * perlin_noise_2d(
            resolution_factor=resolution_factor,
            shape=shape,
            density=(frequency*density[0], frequency*density[1]),
            aspect=aspect,
            tileable=tileable,
            normalized=normalized
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise