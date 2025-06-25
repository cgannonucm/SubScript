#!/usr/bin/env python
import numpy as np
import numpy.testing
import h5py
from typing import Callable

from subscript.scripts.spatial import project3d, project2d
from subscript.wrappers import gscript
from subscript.defaults import ParamKeys
from subscript.util import deprecated

def logical_or(arg1: (np.ndarray[bool] | Callable), arg2: (np.ndarray[bool] | Callable)):
    """
    Create a logical OR function from two nodefilters or boolean arrays.

    Parameters
    ----------
    arg1 : np.ndarray of bool or Callable
        First condition array or callable returning a boolean array.
    arg2 : np.ndarray of bool or Callable
        Second condition array or callable returning a boolean array.

    Returns
    -------
    Callable
        A function that returns the element-wise logical OR of `arg1` and `arg2` when called.
    """
    _a1 = arg1
    if isinstance(arg1, np.ndarray):
        _a1 = lambda *a, **k: arg1
    _a2 = arg2
    if isinstance(arg2, np.ndarray):
        _a2 = lambda *a, **k: arg2
    return lambda *a, **k: _a1(*a, **k) | _a2(*a, **k)

@deprecated("Use logical_or() instead")
def nfor(*args, **kwargs):
    """
    Deprecated. Use `logical_or()` instead.
    """
    return logical_or(*args, **kwargs)

def logical_and(arg1: (np.ndarray[bool] | Callable), arg2: (np.ndarray[bool] | Callable)):
    """
    Create a logical AND function from two nodefilters or boolean arrays.

    Parameters
    ----------
    arg1 : np.ndarray of bool or Callable
        First condition array or callable returning a boolean array.
    arg2 : np.ndarray of bool or Callable
        Second condition array or callable returning a boolean array.

    Returns
    -------
    Callable
        A function that returns the element-wise logical AND of `arg1` and `arg2` when called.
    """
    _a1 = arg1
    if isinstance(arg1, np.ndarray):
        _a1 = lambda *a, **k: arg1
    _a2 = arg2
    if isinstance(arg2, np.ndarray):
        _a2 = lambda *a, **k: arg2
    return lambda *a, **k: _a1(*a, **k) & _a2(*a, **k)


@deprecated("Use logical_and() instead")
def nfand(*args, **kwargs):
    """
    Deprecated. Use `logical_and()` instead.
    """
    return logical_and(*args, **kwargs)

def logical_not(arg: (np.ndarray[bool] | Callable)):
    """
    Create a logical NOT function from a nodefilter or boolean array.

    Parameters
    ----------
    arg : np.ndarray of bool or Callable
        Condition array or callable returning a boolean array.

    Returns
    -------
    Callable
        A function that returns the element-wise logical NOT of `arg` when called.
    """
    _a1 = arg
    if isinstance(arg, np.ndarray):
        _a1 = lambda *a, **k: arg
    return lambda *a, **k: np.logical_not(_a1(*a, **k))

@deprecated("Use logical_not() instead")
def nfnot(*args, **kwargs):
    """
    Deprecated. Use `logical_not()` instead.
    """
    return logical_not(*args, **kwargs)

@gscript
def allnodes(gout, **kwargs):
    """
    Return a boolean array selecting all nodes.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.

    Returns
    -------
    np.ndarray of bool
        Array of `True` values with the same shape as any value in `gout`.
    """
    return np.ones(gout[next(iter(gout))].shape, dtype=bool)

@deprecated("Use allnodes() instead")
def nfilter_all(*args, **kwargs):
    """
    Deprecated. Use `allnodes()` instead.
    """
    return allnodes(*args, **kwargs)

@gscript
def hosthalos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs):
    """
    Select only host halos (i.e., isolated halos).

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    key_is_isolated : str
        Key used to identify isolated halos.

    Returns
    -------
    np.ndarray of bool
        Boolean mask selecting host (isolated) halos.
    """
    return (gout[key_is_isolated] == 1)

@deprecated("Use hosthalos() instead")
def nfilter_halos(*args, **kwargs):
    """
    Deprecated. Use `hosthalos()` instead.
    """
    return hosthalos(*args, **kwargs)

@gscript
def subhalos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs):
    """
    Select only subhalos (i.e., non-isolated halos).

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    key_is_isolated : str
        Key used to identify isolated halos.

    Returns
    -------
    np.ndarray of bool
        Boolean mask selecting subhalos.
    """
    return (gout[key_is_isolated] == 0)

@deprecated("Use subhalos() instead")
def nfilter_subhalos(*args, **kwargs):
    """
    Deprecated. Use `subhalos()` instead.
    """
    return subhalos(*args, **kwargs)

@gscript
def interval(gout, min, max, key=None, getval=None, inclmin=True, inclmax=False, **kwargs):
    """
    Select nodes within a numerical interval.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    min : float
        Lower bound of the interval.
    max : float
        Upper bound of the interval.
    key : str, optional
        Key in `gout` to apply the interval to.
    getval : Callable, optional
        Function to compute the value array from `gout`.
    inclmin : bool
        Whether to include the lower bound.
    inclmax : bool
        Whether to include the upper bound.

    Returns
    -------
    np.ndarray of bool
        Boolean mask selecting values in the specified range.
    """
    if key is not None:
        val = gout[key]
    if getval is not None:
        val = getval(gout, **kwargs)
    lb = min <= val if inclmin else min < val
    ub = val <= max if inclmax else val < max
    return lb & ub

@deprecated("Use interval() instead")
def nfilter_range(*args, **kwargs):
    """
    Deprecated. Use `interval()` instead.
    """
    return interval(*args, **kwargs)

@gscript
def most_massive_progenitor(gout, key_mass_basic=ParamKeys.mass_basic, **kwargs):
    """
    Select only the most massive progenitor node.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    key_mass_basic : str
        Key for the basic mass quantity.

    Returns
    -------
    np.ndarray of bool
        Boolean mask selecting the node with the maximum mass.
    """
    out = np.logical_not(allnodes(gout,**kwargs))
    immp = np.argmax(gout[key_mass_basic])
    out[immp] = True
    return out

@deprecated("Use most_massive_progenitor() instead")
def nfilter_most_massive_progenitor(*args, **kwargs):
    """
    Deprecated. Use `most_massive_progenitor()` instead.
    """
    return most_massive_progenitor(*args, **kwargs)

@gscript
def withinrv(gout, key_rvir=ParamKeys.rvir, key_mass_basic=ParamKeys.mass_basic, inclusive=True, **kwargs):
    """
    Select subhalos within the virial radius of the most massive progenitor.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    key_rvir : str
        Key for virial radius values.
    key_mass_basic : str
        Key for the basic mass quantity.
    inclusive : bool
        Whether to include nodes exactly at the virial radius.

    Returns
    -------
    np.ndarray of bool
        Boolean mask for nodes within the virial radius.
    """
    fmmp = most_massive_progenitor(gout, key_mass_basic=key_mass_basic, **kwargs)
    rv = gout[key_rvir][fmmp][0]
    return interval(gout, min=0, max=rv, inclmin=True, inclmax=inclusive, getval=project3d)

@deprecated("Use withinrv() instead")
def nfilter_virialized(*args, **kwargs):
    """
    Deprecated. Use `withinrv()` instead.
    """
    return withinrv(*args, **kwargs)

@gscript
def subhalos_valid(gout, mass_min=-np.inf, mass_max=np.inf, key_mass=ParamKeys.mass,
                   kwargs_nfilter_subhalos=None, kwargs_nfilter_virialized=None, kwargs_nfilter_range=None, **kwargs):
    """
    Select subhalos within the virial radius and a given mass range.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    mass_min : float
        Lower bound of the mass range.
    mass_max : float
        Upper bound of the mass range.
    key_mass : str
        Key for the mass quantity.
    kwargs_nfilter_subhalos : dict, optional
        Additional arguments for `subhalos`.
    kwargs_nfilter_virialized : dict, optional
        Additional arguments for `withinrv`.
    kwargs_nfilter_range : dict, optional
        Additional arguments for `interval`.

    Returns
    -------
    np.ndarray of bool
        Boolean mask selecting subhalos within range and inside the virial radius.
    """
    kwargs_nfilter_subhalos   = {} if kwargs_nfilter_subhalos is None else kwargs_nfilter_subhalos
    kwargs_nfilter_virialized = {} if kwargs_nfilter_virialized is None else kwargs_nfilter_virialized
    kwargs_nfilter_range      = {} if kwargs_nfilter_range is None else kwargs_nfilter_range

    a = subhalos(gout, **kwargs_nfilter_subhalos)
    b = withinrv(gout, **kwargs_nfilter_virialized)
    c = interval(gout, min=mass_min, max=mass_max, key=key_mass, **kwargs_nfilter_range)

    return a & b & c

@deprecated("Use subhalos_valid() instead")
def nfilter_subhalos_valid(*args, **kwargs):
    """
    Deprecated. Use `subhalos_valid()` instead.
    """
    return subhalos_valid(*args, **kwargs)

@gscript
def r3d(gout, rmin, rmax, **kwargs):
    """
    Select nodes within a 3D radial interval.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    rmin : float
        Minimum 3D radius.
    rmax : float
        Maximum 3D radius.

    Returns
    -------
    np.ndarray of bool
        Boolean mask for nodes within the 3D radial interval.
    """
    return interval(gout, rmin, rmax, getval=project3d, **kwargs)

@deprecated("Use r3d() instead")
def nfiler_project3d(*args, **kwargs):
    """
    Deprecated. Use `r3d()` instead.
    """
    return r3d(*args, **kwargs)

@gscript
def r2d(gout, rmin, rmax, normvector, **kwargs):
    """
    Select nodes within a 2D projected radial interval.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    rmin : float
        Minimum 2D radius.
    rmax : float
        Maximum 2D radius.
    normvector : array-like
        Normal vector for projection.

    Returns
    -------
    np.ndarray of bool
        Boolean mask for nodes within the 2D projected radial interval.
    """
    return interval(gout, rmin, rmax, getval=project2d, normvector=normvector, **kwargs)

@deprecated("Use r2d() instead")
def nfilter_project2d(*args, **kwargs):
    """
    Deprecated. Use `r2d()` instead.
    """
    return r2d(*args, **kwargs)
