import numpy as np
import h5py
from numpy import testing

from subscript.nfilters import nfilter_virialized
from subscript.defaults import  ParamKeys
def test_nfilter_virialized():
    # Test selecting halos within virial radius
    mockdata = {
                ParamKeys.mass_basic : np.array((5.0, 1.0, 1.0, 1.0, 1.0)),
                ParamKeys.rvir       : np.array((0.5, 0  , 0  , 0  , 0  )),
                ParamKeys.x          : np.array((0  , 1.0, 0.5, 0.2, 0.1)),
                ParamKeys.y          : np.array((0  , 1.0, 0.5, 0  , 0.1)),
                ParamKeys.z          : np.array((0  , 0  , 0.5, 0  , 0.1))
    }
    
    # Create test
    out_rv       = nfilter_virialized(mockdata)
    out_expected = np.array((True, False, False, True, True)) 
    testing.assert_equal(out_rv, out_expected)

