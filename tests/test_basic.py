import h5py
import numpy as np
from numpy import testing

from subscript.tabulatehdf5 import tabulate_trees
from subscript.scripts.basic import nodedata
from subscript.defaults import  ParamKeys
from subscript.nfilters import nfilter_halos


def test_nfilter_virialized():
    # Test script + filter
    path_dmo    = "tests/data/test.hdf5"
    gout        = tabulate_trees(h5py.File(path_dmo))
    #print(nfilter_halos(gout))
    out_nd      = nodedata(gout, (ParamKeys.mass, ParamKeys.z_lastisolated), 
                            nodefilter=nfilter_halos, summarize=True,
                            statfuncs=(np.mean, np.std))
    
    out_nd_flat = np.asanyarray(out_nd).flatten()
    expected    = np.array((1E13, 0.5, 0, 0))
    testing.assert_allclose(out_nd_flat, expected)