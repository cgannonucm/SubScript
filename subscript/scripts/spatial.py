#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py

from subscript.wrappers import gscript, NodeFilterWrapper, nfiltercallwrapper
from subscript.nfilters import nfilter_range
from subscript.defaults import ParamKeys

@gscript
def project3d(gout, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs):
    return np.linalg.norm(np.asarray((gout[key_x], gout[key_y], gout[key_z])), axis=0)

@gscript
def project2d(gout, norm, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs):
    coords = np.asarray((gout[key_x], gout[key_y], gout[key_z]))
    # Projection equations, just pythagorean theorem
    # r2^2 = (|r|)^2 + (r.un)^2
    rnorm  = np.linalg.norm(coords, axis=0)
    rdotun = np.dot(coords, norm / np.linalg.norm(norm))
    return np.sqrt(rdotr**2 - rdotun**2)

@gscripts
def _nfilter_project_3d(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, rmin, rmax, **kwargs):
        return nfilter_range(gout, rmin, rmax, get_val=project3d, **kwargs)

@gscripts
def _nfilter_project_2d(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, rmin, rmax, norm, **kwargs):
        return nfilter_range(gout, rmin, rmax, get_val=project2d, norm=norm, **kwargs)

nfilter_project_3d = _nfilter_project_3d()
nfilter_project_2d = _nfilter_project_2d()

def main():
    path_dmo = "../../data/test.hdf5"
    gout = h5py.File(path_dmo)

    print(project3d(gout))

if __name__ == "__main__":
    main()
