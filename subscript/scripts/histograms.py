#!/usr/bin/env python
import numpy as np
from subscript.wrappers import gscript
from subscript.defaults import ParamKeys

def bin_avg(bins):
    return (bins[:-1] + bins[1:]) / 2

@gscript
def hist(gout, key_hist=None, getval=None, bins=None, range=None, density=False, weights=None, kwargs_massfunction_hist = None, **kwargs):
    """Wrapper for np.histogram, provide a key or a function"""
    kwargs_massfunction_hist = {} if kwargs_massfunction_hist is None else kwargs_massfunction_hist
    if key_hist is not None:
        a = gout[key_hist]
    if getval is not None:
        a = getval(gout, **kwargs)
    return np.histogram(a, bins=bins, range=range, density=density, weights=weights, **kwargs_massfunction_hist)

@gscript
def massfunction(gout, key_mass=ParamKeys.mass, bins=None, range=None, **kwargs): 
    _hist, _bins = hist(gout, key_hist=key_mass, bins=bins, range=range)
    return _hist / bin_avg(bins), _bins 

def main():
    import h5py
    from subscript.tabulatehdf5 import tabulate_trees
    from subscript.wrappers import nodedata 
    from subscript.nfilters import nfilter_subhalos_valid


    # Test script + filter
    path_dmo    = "../../data/test.hdf5"
    gout        = tabulate_trees(h5py.File(path_dmo))

    _statfuncs = (np.mean, np.std)
    #print(hist(gout, ParamKeys.mass_basic, bins=np.logspace(8,13, 5), summarize=False))
    #print(hist(gout, ParamKeys.mass_basic, bins=np.logspace(8,13, 5), summarize=True, statfuncs=_statfuncs))
    _filter = nfilter_subhalos_valid.freeze(mass_min=1E8, mass_max=1E13, key_mass=ParamKeys.mass_bound)
    print(hist(gout, ParamKeys.mass_basic, bins=np.logspace(8, 13, 5), summarize=True, nodefilter=_filter, statfuncs=_statfuncs))
    print(massfunction(gout, ParamKeys.mass_basic, bins=np.logspace(8, 13, 5), summarize=True, nodefilter=_filter, statfuncs=_statfuncs))

if __name__ == "__main__": 
    main()