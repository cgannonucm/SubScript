#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py
from typing import Callable

from subscript.scripts.spatial import project3d, project2d
from subscript.wrappers import gscript
from subscript.defaults import ParamKeys

# This design is chosen to allow for lsps 
# basically impossible impossible to get 
# type hints unless we design our function like this
def nfor(arg1:(np.ndarray[bool] | Callable), arg2:(np.ndarray[bool] | Callable)):
    _a1 = arg1
    if isinstance(arg1, np.ndarray):
        _a1 = lambda *a, **k: arg1
    _a2 = arg2
    if isinstance(arg2, np.ndarray):
        _a2 = lambda *a, **k: arg2
    return lambda *a, **k: _a1(*a, **k) | _a2(*a, **k)

def nfand(arg1:(np.ndarray[bool] | Callable), arg2:(np.ndarray[bool] | Callable)):
    _a1 = arg1
    if isinstance(arg1, np.ndarray):
        _a1 = lambda *a, **k: arg1
    _a2 = arg2
    if isinstance(arg2, np.ndarray):
        _a2 = lambda *a, **k: arg2
    return lambda *a, **k: _a1(*a, **k) & _a2(*a, **k)

def nfnot(arg:(np.ndarray[bool] | Callable)):
    _a1 = arg
    if isinstance(arg, np.ndarray):
        _a1 = lambda *a, **k: arg
    return lambda *a, **k: np.logical_not(_a1(*a, **k)) 

@gscript
def nfilter_all(gout, **kwargs):
    return np.ones(gout[next(iter(gout))].shape, dtype=bool)

@gscript
def nfilter_halos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs):
    return (gout[key_is_isolated] == 1)

@gscript
def nfilter_subhalos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs):
    return (gout[key_is_isolated] == 0)

@gscript
def nfilter_range(gout, min, max, key = None, getval = None, inclmin = True, inclmax = False, **kwargs):
    if key is not None:
        val = gout[key]
    if getval is not None: 
        val = getval(gout, **kwargs)
    lb = min <= val if inclmin else min < val
    ub = val <= max if inclmin else val < max
    return lb & ub

@gscript
def nfilter_most_massive_progenitor(gout, key_mass_basic=ParamKeys.mass_basic, **kwargs):
    out = np.logical_not(nfilter_all(gout,**kwargs))
    immp = np.argmax(gout[key_mass_basic])
    out[immp] = True
    return out

@gscript
def nfilter_virialized(gout, key_rvir=ParamKeys.rvir, key_mass_basic=ParamKeys.mass_basic, inclusive = True, **kwargs):
    fmmp = nfilter_most_massive_progenitor(gout, key_mass_basic=key_mass_basic, **kwargs)
    rv = gout[key_rvir][fmmp][0]
    return nfilter_range(gout, min=0, max=rv, inclmin=True, inclmax=inclusive, getval=project3d)

@gscript
def _nfilter_subhalos_valid(gout, mass_min, mass_max, key_mass=ParamKeys.mass, 
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

@gscript
def nfilter_project_3d(gout, rmin, rmax, **kwargs):
    return nfilter_range(gout, rmin, rmax, getval=project3d, **kwargs)

@gscript
def nfilter_project_2d(gout, rmin, rmax, norm, **kwargs):
    return nfilter_range(gout, rmin, rmax, getval=project2d, norm=norm, **kwargs)

