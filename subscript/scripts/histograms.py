#!/usr/bin/env python
import numpy as np
from subscript.wrappers import gscript, gscript_proj
from subscript.defaults import ParamKeys
from subscript.scripts.spatial import project3d, project2d

def bin_avg(bins):
    """
    Compute the average value of each bin.

    Parameters
    ----------
    bins : array_like
        Array of bin edges.

    Returns
    -------
    ndarray
        Array of average values for each bin.
    """
    return (bins[1:] + bins[:-1] ) / 2

def bin_size(bins):
    """
    Compute the width of each bin.

    Parameters
    ----------
    bins : array_like
        Array of bin edges.

    Returns
    -------
    ndarray
        Array of bin widths.
    """
    return (bins[1:] - bins[:-1])

@gscript
def hist(gout, key_hist=None, getval=None, bins=None, range=None, density=False, weights=None, kwargs_hist = None, **kwargs):
    """
    Compute a histogram using data extracted from Galacticus-like output.

    Parameters
    ----------
    gout : NodeProperties
        Dictionary-like structure containing Galacticus output node data.
    key_hist : str, optional
        Key to extract data from `gout` for histogramming.
    getval : callable, optional
        Function that extracts or computes the histogram data from `gout`.
    bins : int or sequence of scalars or str, optional
        If bins is an int, it defines the number of equal-width bins.
        If a sequence, it defines the bin edges. If a string, it defines
        a method for automatic binning.
    range : tuple or None, optional
        The lower and upper range of the bins.
    density : bool, default=False
        If True, the result is the value of the probability density function.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.
    kwargs_hist : dict, optional
        Additional keyword arguments for `np.histogram`.

    Returns
    -------
    hist : ndarray
        The values of the histogram.
    bin_edges : ndarray
        Return the bin edges (length(hist)+1).
    """
    kwargs_hist = {} if kwargs_hist is None else kwargs_hist
    if key_hist is not None:
        a = gout[key_hist]
    if getval is not None:
        a = getval(gout, **kwargs)
    return np.histogram(a, bins=bins, range=range, density=density, weights=weights)

@gscript
def massfunction(gout, key_mass=ParamKeys.mass, bins=None, range=None, **kwargs):
    """
    Compute the halo mass function as number density per mass bin width.

    Parameters
    ----------
    gout : NodeProperties
        Dictionary-like structure containing Galacticus output node data.
    key_mass : str, default=ParamKeys.mass
        Key for the mass property in `gout`.
    bins : int or sequence of scalars or str, optional
        Bin specification passed to `np.histogram`.
    range : tuple or None, optional
        Lower and upper range for the bins.
    **kwargs
        Additional arguments passed to the `hist` function.

    Returns
    -------
    mass_function : ndarray
        Number density per mass bin width.
    bin_edges : ndarray
        The edges of the mass bins.
    """
    _hist, _bins = hist(gout, key_hist=key_mass, bins=bins, range=range, **kwargs)
    return _hist / bin_size(_bins), _bins 

@gscript
def spatial3d_dn(gout, bins=None, range=None, kwargs_hist = None, **kwargs):
    """
    Compute the 3D radial number density distribution.

    Parameters
    ----------
    gout : NodeProperties
        Dictionary-like structure containing Galacticus output node data.
    bins : int or sequence of scalars or str, optional
        Bin specification passed to `np.histogram`.
    range : tuple or None, optional
        Lower and upper range for the bins.
    kwargs_hist : dict, optional
        Additional arguments for `np.histogram`.
    **kwargs
        Additional arguments passed to `project3d`.

    Returns
    -------
    hist : ndarray
        Number of objects in each radial bin.
    bin_edges : ndarray
        The edges of the radial bins.
    """
    r = project3d(gout, **kwargs) 
    return np.histogram(r, bins=bins, range=range)

@gscript
def spatial3d_dndv(gout, bins=None, range=None, kwargs_hist = None, **kwargs):
    """
    Compute the 3D radial number density per unit volume.

    Parameters
    ----------
    gout : NodeProperties
        Dictionary-like structure containing Galacticus output node data.
    bins : int or sequence of scalars or str, optional
        Bin specification passed to `np.histogram`.
    range : tuple or None, optional
        Lower and upper range for the bins.
    kwargs_hist : dict, optional
        Additional arguments for `np.histogram`.

    Returns
    -------
    dndv : ndarray
        Number density per unit volume in each bin.
    bin_edges : ndarray
        The edges of the radial bins.
    """
    dn, dn_r = spatial3d_dn(gout, bins=bins, range=range, kwargs_hist=kwargs_hist, **kwargs)
    dv = 4 / 3 * np.pi * (dn_r[1:]**3 - dn_r[:-1]**3)
    return dn / dv, dn_r

@gscript_proj
def spatial2d_dn(gout, normvector, bins=None, range=None, kwargs_hist = None, **kwargs):
    """
    Compute the 2D projected number density distribution.

    Parameters
    ----------
    gout : NodeProperties
        Dictionary-like structure containing Galacticus output node data.
    normvector : array_like
        Normal vector defining the projection axis.
    bins : int or sequence of scalars or str, optional
        Bin specification passed to `np.histogram`.
    range : tuple or None, optional
        Lower and upper range for the bins.
    kwargs_hist : dict, optional
        Additional arguments for `np.histogram`.
    **kwargs
        Additional arguments passed to `project2d`.

    Returns
    -------
    hist : ndarray
        Number of objects in each projected radial bin.
    bin_edges : ndarray
        The edges of the radial bins.
    """
    r = project2d(gout,normvector=normvector, **kwargs) 
    return np.histogram(r, bins=bins, range=range)

@gscript_proj
def spatial2d_dnda(gout, normvector, bins=None, range=None, kwargs_hist = None, **kwargs):
    """
    Compute the 2D projected number density per unit area.

    Parameters
    ----------
    gout : NodeProperties
        Dictionary-like structure containing Galacticus output node data.
    normvector : array_like
        Normal vector defining the projection axis.
    bins : int or sequence of scalars or str, optional
        Bin specification passed to `np.histogram`.
    range : tuple or None, optional
        Lower and upper range for the bins.
    kwargs_hist : dict, optional
        Additional arguments for `np.histogram`.
    **kwargs
        Additional arguments passed to `project2d`.

    Returns
    -------
    dnda : ndarray
        Number density per unit area in each bin.
    bin_edges : ndarray
        The edges of the projected radial bins.
    """
    dn, dn_r = spatial2d_dn(gout, normvector, bins=bins, range=range, kwargs_hist = kwargs_hist, **kwargs)
    da = np.pi * (dn_r[1:]**2 - dn_r[:-1]**2)
    return dn / da, dn_r
