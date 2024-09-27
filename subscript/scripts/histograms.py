#!/usr/bin/env python
import numpy as np
from subscript.wrappers import gscript, gscript_proj
from subscript.defaults import ParamKeys
from subscript.scripts.spatial import project3d, project2d

def bin_avg(bins):
    return (bins[1:] + bins[:-1] ) / 2

def bin_size(bins):
    return (bins[1:] - bins[:-1])

@gscript
def hist(gout, key_hist=None, getval=None, bins=None, range=None, density=False, weights=None, kwargs_hist = None, **kwargs):
    """Wrapper for np.histogram, provide a key or a function"""
    kwargs_hist = {} if kwargs_hist is None else kwargs_hist
    if key_hist is not None:
        a = gout[key_hist]
    if getval is not None:
        a = getval(gout, **kwargs)
    return np.histogram(a, bins=bins, range=range, density=density, weights=weights)

@gscript
def massfunction(gout, key_mass=ParamKeys.mass, bins=None, range=None, **kwargs): 
    _hist, _bins = hist(gout, key_hist=key_mass, bins=bins, range=range)
    return _hist / bin_size(_bins), _bins 

@gscript
def spatial3d_dn(gout, bins=None, range=None, kwargs_hist = None, **kwargs):
    r = project3d(gout, **kwargs) 
    return np.histogram(r, bins=bins, range=range)

@gscript
def spatial3d_dndv(gout, bins=None, range=None, kwargs_hist = None, **kwargs): 
    dn, dn_r = spatial3d_dn(gout, bins=bins, range=range, kwargs_hist=kwargs_hist)
    dv = 4 / 3 * np.pi * (dn_r[1:]**3 - dn_r[:-1]**3)
    return dn / dv, dn_r

@gscript_proj
def spatial2d_dn(gout, normvector, bins=None, range=None, kwargs_hist = None, **kwargs):
    r = project2d(gout,normvector=normvector, **kwargs) 
    return np.histogram(r, bins=bins, range=range)

@gscript_proj
def spatial2d_dnda(gout, normvector, bins=None, range=None, kwargs_hist = None, **kwargs):
    dn, dn_r = spatial2d_dn(gout, normvector, bins=bins, range=range, kwargs_hist = kwargs_hist, **kwargs)
    da = np.pi * (dn_r[1:]**2 - dn_r[:-1]**2)
    return dn / da, dn_r