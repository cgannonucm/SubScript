#!/usr/bin/env python
from __future__ import annotations
from typing import Any, Callable, Iterable, List
from collections import UserDict
from functools import reduce
import numpy as np
from numpy.typing import ArrayLike
import h5py
from copy import copy

from subscript.util import is_arraylike
from subscript.tabulatehdf5 import NodeProperties, tabulate_trees
from subscript import tabulatehdf5
from subscript import defaults

def reduce_input(l, out=None):
    """
    Recursively flatten a nested list structure and collect dictionaries.

    Parameters
    ----------
    l : list
        Nested list potentially containing dictionaries or UserDict objects.
    out : list, optional
        List to accumulate dictionaries. If None, a new list is created.

    Returns
    -------
    list of dict or UserDict
        Flattened list containing all dictionaries/UserDicts found in `l`.

    """
    if out is None:
        out = []
    
    for i in l:
        if isinstance(i, (dict, UserDict)):
            out.append(i)
            continue
        reduce_input(i, out)
    return out
 
def format_nodedata(gout, out_index=-1)->Iterable[NodeProperties]:
    """
    Convert various Galacticus-like outputs into a list of NodeProperties objects.

    Parameters
    ----------
    gout : h5py.File or dict or UserDict or iterable
        Galacticus-like output data, such as an HDF5 file, a node property dictionary,
        or an iterable of such objects.
    out_index : int, optional
        Index of the output to extract when `gout` is an HDF5 file (default is -1).

    Returns
    -------
    iterable of NodeProperties
        A list of NodeProperties wrapping the processed input data.

    Raises
    ------
    RuntimeError
        If the type of `gout` is not recognized.
    """
    if isinstance(gout, (dict, UserDict)):
        _gout = [NodeProperties(gout), ]
    elif isinstance(gout, h5py.File):
        _gout = tabulate_trees(gout, out_index=out_index)
    elif isinstance(gout, Iterable): 
        _gout = reduce_input([format_nodedata(o, out_index=out_index) for o in gout])
    else:
        raise RuntimeError(f"Unrecognized data type for gout {type(gout)}")
    return _gout

def _summarize(outs, summarize, statfuncs):
    if summarize is None or not summarize:
        return outs

    _statfuncs = [np.mean, ] if statfuncs is None else statfuncs

    if isinstance(outs[0], Iterable):
        eval_stats = lambda f,m: f(np.asarray([treeo[m] for treeo in outs]), axis=0)
        summary = [[eval_stats(f,m) for m, _ in enumerate(outs[0])] for f in _statfuncs]
    else:
        eval_stats = lambda f: f(np.asarray([treeo for treeo in outs]), axis=0)
        summary = [eval_stats(f) for f in _statfuncs]
    return summary


# Eliminate lists of 1 item recursively
def _format_out(o):
    if (not isinstance(o, Iterable)) or (isinstance(o, str)):
        return o
    if len(o) == 1:
        return _format_out(o[0])
    out = [_format_out(i) for i in o]
    if isinstance(o, np.ndarray):
        return np.asarray(out)
    return out

def gscript(func):
    """
    Decorator to wrap a function that processes Galacticus-like node data.

    This decorator handles input data formatting, node filtering, multiple tree realizations,
    and optional summary statistics computation.

    Parameters
    ----------
    func : callable
        The function to be wrapped. Must accept a filtered NodeProperties object
        as the first argument as well as an arbitrary number of key word arguments.

    Returns
    -------
    callable
        Wrapped function that accepts:

        gout : h5py.File, NodeProperties, or dict
            Dictionary-like structure containing Galacticus output node data.
        *args : tuple
            Additional positional arguments passed to `func`.
        nfilter : callable or array-like of bool, optional
            Node filter applied before passing data to `func`.
        summarize : bool, optional
            If True, return summary statistics over all trees.
        statfuncs : iterable of callables, optional
            Functions to compute summary statistics (default: [np.mean]).
        out_index : int, optional
            Output index to extract if `gout` is an HDF5 file.
        **kwargs : dict
            Additional keyword arguments passed to `func`.

    Notes
    -----
    If `gout` is None, returns a partially applied wrapper function to be called later.
    """
    def wrap(gout:(h5py.File | NodeProperties | dict), 
                *args, 
                nfilter:(Callable | np.ndarray[bool])=None, 
                summarize:bool=False, 
                statfuncs:Iterable[Callable] = None,
                out_index:int=-1,
                **kwargs):         
        if gout is None:
            _kwargs = dict(
                           nfilter=nfilter, 
                           summarize=summarize,
                           statfuncs=statfuncs,
                           out_index=out_index,
                          )

            return lambda gout, **k: wrap(
                                          gout,
                                          *args,
                                          **(_kwargs | kwargs | k)
                                         )
        
        outs = []         
        trees = format_nodedata(gout, out_index)

        for nodestree in trees:
            _nodestree = nodestree.unfilter()

            if nfilter is None:
                _nodefilter = None
            elif isinstance(nfilter, Callable):
                _nodefilter = nfilter(_nodestree, **kwargs)
            elif is_arraylike(nfilter):
                _nodefilter = np.asarray(nfilter, dtype=bool)
            else:
                TypeError("Unrecognized type provided to nodefilter")

            _nodestree_filtered = _nodestree.filter(_nodefilter)

            o = func(_nodestree_filtered, *args, **(kwargs | dict(nfilter=_nodefilter)))
            single_out = isinstance(o, np.ndarray) 
            _o = [o,] if single_out else o
            outs.append(_o)

        summary = _summarize(outs, summarize=summarize, statfuncs=statfuncs)

        return _format_out(summary)
    return wrap

def gscript_proj(func):
    """
    Decorator for functions that involve projection with one or multiple normal vectors.

    This decorator allows passing multiple projection vectors, treating each as a separate
    realization (tree) and processing them accordingly.

    Parameters
    ----------
    func : callable
        Function to wrap. Must accept a `normvector` argument representing the projection vector.

    Returns
    -------
    callable
        Wrapped function that accepts:

        gout : h5py.File, NodeProperties, or dict
            Dictionary-like structure containing Galacticus output node data.
        normvector : array-like
            One or more normal vectors used for projection.
        *args : tuple
            Additional positional arguments passed to `func`.
        **kwargs : dict
            Additional keyword arguments passed to `func`.

    Notes
    -----
    If `normvector` is a 2D array, each vector is treated as a separate tree and processed individually.
    """
    def wrap(gout, normvector, *args, **kwargs):
        normvector = np.asarray(normvector)

        if normvector.ndim == 1:
            #raise NotImplementedError()
            return gscript(func)(gout, *args, normvector=normvector, **kwargs)
        if normvector.ndim > 2 or normvector.ndim <= 0:
            raise RuntimeError(f'"normvector" must be either 1 or 2 dimensional')

        @gscript
        def wrap_inner_main(gout, **kwargs):
            n = gout.unfilter()['__custom_proj_iter__'][0]
            return func(gout, *args, normvector=normvector[n], **kwargs)


        ## This wrapped as well so we can call with None
        def wrap_inner(gout, **kwargs2):
            _gout = format_nodedata(gout)

            _input = []
            for n, _ in enumerate(normvector):
                _in = copy(_gout)
                for i in _in:
                    _i = copy(i)
                    _i.data['__custom_proj_iter__'] = n * np.ones(i.data[next(i.data.__iter__())].shape[0], dtype=int)
                    _input.append(_i)

            return wrap_inner_main(_input, **(kwargs | kwargs2))

        if gout is None:
            return wrap_inner

        return wrap_inner(gout)

    return wrap

def freeze(func, **kwargs):
    """
    Return a new function with fixed keyword arguments applied.

    Parameters
    ----------
    func : callable
        The function to partially apply keyword arguments to.
    **kwargs : dict
        Keyword arguments to fix in the returned function.

    Returns
    -------
    callable
        A new function that calls `func` with the given `kwargs` fixed.
    """
    return lambda gout, *a, **k: func(gout, *a, **(k | kwargs))

def multiproj(func, nfilter):
    """
    Apply `gscript_proj` decorator to a function with a fixed node filter.

    Parameters
    ----------
    func : callable
        Function to wrap, expected to accept a node filter.
    nfilter : callable or array-like of bool
        Node filter to be fixed for the wrapped function.

    Returns
    -------
    callable
        Function wrapped with `gscript_proj` and the node filter applied.
    """
    return gscript_proj(freeze(func, nfilter=nfilter))
