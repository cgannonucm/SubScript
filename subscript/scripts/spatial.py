#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py

from subscript.wrappers import gscript
from subscript.defaults import ParamKeys

@gscript
def project3d(gout, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs):
    return np.linalg.norm(np.asarray((gout[key_x], gout[key_y], gout[key_z])), axis=0)


def main():
    path_dmo = "../../data/test.hdf5"
    gout = h5py.File(path_dmo)

    print(project3d(gout))

if __name__ == "__main__":
    main()
