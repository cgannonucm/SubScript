#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py

from subscript.scripts.spatial import project3d
from subscript.wrappers import NodeFilterWrapper, nfiltercallwrapper
from subscript.defaults import ParamKeys

# This design is chosen to allow for lsps 
# basically impossible impossible to get 
# type hints unless we design our function like this
class _nfilter_all(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, **kwargs):
        return np.ones(gout[next(iter(gout))].shape, dtype=bool)

class _nfilter_halos(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, **kwargs):
        return (gout[ParamKeys.is_isolated] == 1)

class _nfilter_range(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, min, max, key = None, getval = None, inclmin = True, inclmax = False, **kwargs):
        if key is not None:
            val = gout[key]
        if getval is not None: 
            val = getval(gout, **kwargs)
        lb = min <= val if inclmin else min < val
        ub = val >= max if inclmin else val > max
        return lb & ub

class _nfilter_most_massive_progenitor(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, key_mass_basic=ParamKeys.mass_basic, **kwargs):
        out = np.logical_not(nfilter_all(gout,**kwargs))
        immp = np.argmax(gout[key_mass_basic])
        out[immp] = True
        return out

class _nfilter_select_virialized(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, key_rvir=ParamKeys.rvir, inclusive = True, **kwargs):
        fmmp = nfilter_most_massive_progenitor(gout, **kwargs)
        rv = gout[key_rvir][fmmp][0]
        return nfilter_range(gout, min=0, max=rv, inclmin=True, inclmax=inclusive, getval=project3d)


nfilter_halos                   =  _nfilter_halos                  ()
nfilter_subhalos                = ~nfilter_halos
nfilter_all                     =  _nfilter_all                    ()
nfilter_range                   =  _nfilter_range                  ()
nfilter_most_massive_progenitor =  _nfilter_most_massive_progenitor()
nfilter_select_virialized       =  _nfilter_select_virialized      ()

def main():
    from subscript.tabulatehdf5 import tabulate_trees
    from subscript.wrappers import nodedata

    path_dmo = "../data/test.hdf5"
    gout = tabulate_trees(h5py.File(path_dmo))
    #print(nfilter_halos(gout))
    out = nodedata(gout, (ParamKeys.mass, ParamKeys.z_lastisolated), 
                        nodefilter=nfilter_halos, summarize=True,
                        statfuncs=(np.mean, np.std))
    
    out_flat = np.asanyarray(out).flatten()
    expected = np.array((1E13, 0.5, 0, 0))
    np.testing.assert_allclose(out_flat, expected)

    # Create test
    print(nfilter_select_virialized(gout))

if __name__ == "__main__": 
    main()