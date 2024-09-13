#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py

from subscript.scripts.spatial import project3d, project2d
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
    def __call__(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs):
        return (gout[key_is_isolated] == 1)

class _nfilter_range(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, min, max, key = None, getval = None, inclmin = True, inclmax = False, **kwargs):
        if key is not None:
            val = gout[key]
        if getval is not None: 
            val = getval(gout, **kwargs)
        lb = min <= val if inclmin else min < val
        ub = val <= max if inclmin else val < max
        return lb & ub

class _nfilter_most_massive_progenitor(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, key_mass_basic=ParamKeys.mass_basic, **kwargs):
        out = np.logical_not(nfilter_all(gout,**kwargs))
        immp = np.argmax(gout[key_mass_basic])
        out[immp] = True
        return out

class _nfilte_virialized(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, key_rvir=ParamKeys.rvir, key_mass_basic=ParamKeys.mass_basic, inclusive = True, **kwargs):
        fmmp = nfilter_most_massive_progenitor(gout, key_mass_basic=key_mass_basic, **kwargs)
        rv = gout[key_rvir][fmmp][0]
        return nfilter_range(gout, min=0, max=rv, inclmin=True, inclmax=inclusive, getval=project3d)

class _nfilter_subhalos_valid(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, mass_min, mass_max, key_mass=ParamKeys.mass, 
                    kwargs_nfilter_subhalos = None, kwargs_nfilter_virialized=None, kwargs_nfilter_range=None, 
                    **kwargs):
        """
        A combined nodefilter that
        a) excludes host halos
        b) excludes subhalos beyond the virial radius
        c) selects for subhalos in a given mass bin
        """
        kwargs_nfilter_subhalos   = {} if kwargs_nfilter_subhalos is None else kwargs_nfilter_subhalos
        kwargs_nfilter_virialized = {} if kwargs_nfilter_virialized is None else kwargs_nfilter_virialized
        kwargs_nfilter_range      = {} if kwargs_nfilter_range is None else kwargs_nfilter_range

        a = nfilter_subhalos  (gout, **kwargs_nfilter_subhalos) 
        b = nfilter_virialized(gout, **kwargs_nfilter_virialized) 
        c = nfilter_range     (gout, min=mass_min, max=mass_max, key=key_mass, **kwargs_nfilter_range)

        return a & b & c

class _nfilter_project_3d(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, rmin, rmax, **kwargs):
        return nfilter_range(gout, rmin, rmax, getval=project3d, **kwargs)

class _nfilter_project_2d(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(gout, rmin, rmax, norm, **kwargs):
        return nfilter_range(gout, rmin, rmax, getval=project2d, norm=norm, **kwargs)


nfilter_halos                   =  _nfilter_halos                  ()
nfilter_subhalos                = ~nfilter_halos
nfilter_all                     =  _nfilter_all                    ()
nfilter_range                   =  _nfilter_range                  ()
nfilter_most_massive_progenitor =  _nfilter_most_massive_progenitor()
nfilter_virialized              =  _nfilte_virialized              ()
nfilter_subhalos_valid          =  _nfilter_subhalos_valid         ()
nfilter_project_3d              = _nfilter_project_3d              ()
nfilter_project_2d              = _nfilter_project_2d              ()

def main():
    from subscript.tabulatehdf5 import tabulate_trees
    from subscript.wrappers import nodedata

    # Test script + filter
    path_dmo    = "../data/test.hdf5"
    gout        = tabulate_trees(h5py.File(path_dmo))
    #print(nfilter_halos(gout))
    out_nd      = nodedata(gout, (ParamKeys.mass, ParamKeys.z_lastisolated), 
                            nodefilter=nfilter_halos, summarize=True,
                            statfuncs=(np.mean, np.std))
    
    out_nd_flat = np.asanyarray(out_nd).flatten()
    expected    = np.array((1E13, 0.5, 0, 0))
    np.testing.assert_allclose(out_nd_flat, expected)


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
    np.testing.assert_equal(out_rv, out_expected)

if __name__ == "__main__": 
    main()