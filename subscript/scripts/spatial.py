#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py

from subscript.wrappers import gscript, gscript_proj
from subscript.defaults import ParamKeys

@gscript
def project3d(gout, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs):
    return np.linalg.norm(np.asarray((gout[key_x], gout[key_y], gout[key_z])), axis=0)

@gscript_proj
def project2d(gout, normvector, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs):
    coords = np.asarray((gout[key_x], gout[key_y], gout[key_z]))
    # Projection equations, just pythagorean theorem
    # r2^2 = (|r|)^2 + (r.un)^2
    rnorm  = np.linalg.norm(coords, axis=0)
    rdotun = np.dot(normvector / np.linalg.norm(normvector), coords)
    return np.sqrt(rnorm**2 - rdotun**2)