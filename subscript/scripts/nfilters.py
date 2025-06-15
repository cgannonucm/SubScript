#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py
from typing import Callable

from subscript.scripts.spatial import project3d, project2d
from subscript.wrappers import gscript
from subscript.defaults import ParamKeys
from subscript.util import deprecated

# This design is chosen to allow for lsps 
# basically impossible impossible to get 
# type hints unless we design our function like this
def logical_or(arg1:(np.ndarray[bool] | Callable), arg2:(np.ndarray[bool] | Callable)):
    _a1 = arg1
    if isinstance(arg1, np.ndarray):
        _a1 = lambda *a, **k: arg1
    _a2 = arg2
    if isinstance(arg2, np.ndarray):
        _a2 = lambda *a, **k: arg2
    return lambda *a, **k: _a1(*a, **k) | _a2(*a, **k)

@deprecated("Use logical_or() instead")
def nfor(*args, **kwargs):
    return logical_or(*args, **kwargs)

def logical_and(arg1:(np.ndarray[bool] | Callable), arg2:(np.ndarray[bool] | Callable)):
    _a1 = arg1
    if isinstance(arg1, np.ndarray):
        _a1 = lambda *a, **k: arg1
    _a2 = arg2
    if isinstance(arg2, np.ndarray):
        _a2 = lambda *a, **k: arg2
    return lambda *a, **k: _a1(*a, **k) & _a2(*a, **k)


@deprecated("Use logical_and() instead")
def nfand(*args, **kwargs):
    return logical_and(*args, **kwargs)


def logical_not(arg:(np.ndarray[bool] | Callable)):
    _a1 = arg
    if isinstance(arg, np.ndarray):
        _a1 = lambda *a, **k: arg
    return lambda *a, **k: np.logical_not(_a1(*a, **k))

@deprecated("Use logical_not() instead")
def nfnot(*args, **kwargs):
    return logical_not(*args, **kwargs)

@gscript
def none(gout, **kwargs):
    return np.ones(gout[next(iter(gout))].shape, dtype=bool)


@deprecated("Use none() instead")
def nfiler_all(*args, **kwargs):
    return none(*args, **kwargs)

@gscript
def hosthalos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs):
    return (gout[key_is_isolated] == 1)


@deprecated("Use hosthalos()  instead")
def nfilter_halos(*args, **kwargs):
    return hosthalos(*args, **kwargs)

@gscript
def subhalos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs):
    return (gout[key_is_isolated] == 0)


@deprecated("Use subhalos() instead")
def nfilter_subhalos(*args, **kwargs):
    return subhalos(*args, **kwargs)

@gscript
def interval(gout, min, max, key = None, getval = None, inclmin = True, inclmax = False, **kwargs):
    if key is not None:
        val = gout[key]
    if getval is not None: 
        val = getval(gout, **kwargs)
    lb = min <= val if inclmin else min < val
    ub = val <= max if inclmin else val < max
    return lb & ub

@deprecated("Use interval() instead")
def nfilter_range(*args, **kwargs):
    return interval(*args, **kwargs)

@gscript
def most_massive_progenitor(gout, key_mass_basic=ParamKeys.mass_basic, **kwargs):
    out = np.logical_not(none(gout,**kwargs))
    immp = np.argmax(gout[key_mass_basic])
    out[immp] = True
    return out


@deprecated("Use most_massive_progenitor() instead")
def nfilter_most_massive_progenitor(*args, **kwargs):
    return most_massive_progenitor(*args, **kwargs)

@gscript
def withinrv(gout, key_rvir=ParamKeys.rvir, key_mass_basic=ParamKeys.mass_basic, inclusive = True, **kwargs):
    fmmp = most_massive_progenitor(gout, key_mass_basic=key_mass_basic, **kwargs)
    rv = gout[key_rvir][fmmp][0]
    return interval(gout, min=0, max=rv, inclmin=True, inclmax=inclusive, getval=project3d)


@deprecated("Use withinrv() instead")
def nfilter_virialized(*args, **kwargs):
    return withinrv(*args, **kwargs)

@gscript
def subhalos_valid(gout, mass_min, mass_max, key_mass=ParamKeys.mass,
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

    a = subhalos  (gout, **kwargs_nfilter_subhalos)
    b = withinrv(gout, **kwargs_nfilter_virialized)
    c = interval     (gout, min=mass_min, max=mass_max, key=key_mass, **kwargs_nfilter_range)

    return a & b & c


@deprecated("Use subhalos_valid() instead")
def nfilter_subhalos_valid(*args, **kwargs):
    return subhalos_valid(*args, **kwargs)

@gscript
def r3d(gout, rmin, rmax, **kwargs):
    return interval(gout, rmin, rmax, getval=project3d, **kwargs)


@deprecated("Use r3d() instead")
def nfiler_project3d(*args, **kwargs):
    return r3d(*args, **kwargs)

@gscript
def r2d(gout, rmin, rmax, normvector, **kwargs):
    return interval(gout, rmin, rmax, getval=project2d, normvector=normvector, **kwargs)

@deprecated("Use r2d() instead")
def nfilter_project2d(*args, **kwargs):
    return r2d(*args, **kwargs)
