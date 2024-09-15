#!/usr/bin/env python
import numpy as np
from subscript.wrappers import gscript
from subscript.defaults import ParamKeys

def bin_avg(bins):
    return (bins[1:] + bins[:-1] ) / 2

def bin_size(bins):
    return (bins[1:] - bins[:-1])

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
    return _hist / bin_size(_bins), _bins 