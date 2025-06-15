#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py

from subscript.wrappers import gscript, gscript_proj
from subscript.defaults import ParamKeys

@gscript
def project3d(gout, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs):
    """
    Compute the 3D Euclidean distance of each node from the origin.

    Parameters
    ----------
    gout : dict of np.ndarray
        Galacticus-like output containing spatial coordinate arrays.
    key_x : str, optional
        Key corresponding to the x-coordinate. Defaults to `ParamKeys.x`.
    key_y : str, optional
        Key corresponding to the y-coordinate. Defaults to `ParamKeys.y`.
    key_z : str, optional
        Key corresponding to the z-coordinate. Defaults to `ParamKeys.z`.

    Returns
    -------
    np.ndarray
        Array of shape (N,) representing the 3D distance of each node from the origin.
    """
    return np.linalg.norm(np.asarray((gout[key_x], gout[key_y], gout[key_z])), axis=0)

@gscript_proj
def project2d(gout, normvector, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs):
    """
    Project the 3D position of each node onto a 2D plane orthogonal to `normvector`.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    normvector : array-like of shape (3,) or shape (m, 3)
        Normal vector defining the projection plane.
        Note: multiple normal vectors can be provided.
    key_x : str, optional
        Key corresponding to the x-coordinate. Defaults to `ParamKeys.x`.
    key_y : str, optional
        Key corresponding to the y-coordinate. Defaults to `ParamKeys.y`.
    key_z : str, optional
        Key corresponding to the z-coordinate. Defaults to `ParamKeys.z`.

    Returns
    -------
    np.ndarray
        Array of shape (N,) representing the projected 2D distance of each node from the origin
        in the plane orthogonal to `normvector`. If multiple normal vectors are provide, then the shape will be
        (m * N, )

    """
    coords = np.asarray((gout[key_x], gout[key_y], gout[key_z]))
    # Projection equations, just pythagorean theorem
    # r2^2 = (|r|)^2 + (r.un)^2
    rnorm  = np.linalg.norm(coords, axis=0)
    rdotun = np.dot(normvector / np.linalg.norm(normvector), coords)
    return np.sqrt(rnorm**2 - rdotun**2)
