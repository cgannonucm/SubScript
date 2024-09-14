#!/usr/bin/env python
from typing import Iterable
from subscript.wrappers import gscript


@gscript
def nodedata(gout, key:(str | Iterable[str]), **kwargs):
    return gout[key]

def main():
    import h5py
    from subscript.defaults import ParamKeys
    path_dmo = "../../data/test.hdf5"
    gout = h5py.File(path_dmo)
    print(nodedata(gout, ParamKeys.mass))

if __name__ == "__main__": 
    main()