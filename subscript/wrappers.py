#!/usr/bin/env python
from __future__ import annotations
from typing import Any, Callable, Iterable
from collections import UserDict
import numpy as np
import h5py

from subscript.tabulatehdf5 import NodeProperties, tabulate_trees
from subscript import tabulatehdf5
from subscript import defaults

def format_nodedata(gout, out_index=-1)->Iterable[NodeProperties]:
    if isinstance(gout, (dict, UserDict)):
        _gout = [NodeProperties(gout), ]
    elif isinstance(gout, h5py.File):
        _gout = tabulate_trees(gout, out_index=out_index)
    elif isinstance(gout, Iterable):
        _gout = [NodeProperties(o) for o in gout]
    else:
        raise RuntimeError(f"Unrecognized data type for gout {type(gout)}")
    return _gout

def gscript(func):
    def wrap(gout:(h5py.File | NodeProperties | dict), *args, 
                nodefilter:(Callable | np.ndarray[bool])=None, summarize:bool=False, statfuncs:Iterable[Callable] = None,
                out_index:int=-1, **kwargs): 
        outs = []         
        trees = format_nodedata(gout, out_index)
        ntrees = len(trees)

        for nodestree in trees:
            _nodestree = nodestree.unfilter()
            _nodefilter = None
            if nodefilter is not None:
                _nodefilter = nodefilter(_nodestree, **kwargs)
            _nodestree_filtered = _nodestree.filter(_nodefilter)
            o = func(_nodestree_filtered, *args, **kwargs)
            single_out = isinstance(o, np.ndarray) 
            _o = [o,] if single_out else o
            outs.append(_o)

        # Eliminate lists of 1 item recursively
        def format_out(o):
            if (not isinstance(o, Iterable)) or (isinstance(o, str)):
                return o
            if len(o) == 1:
                return format_out(o[0])
            out = [format_out(i) for i in o]         
            if isinstance(o, np.ndarray):
                return np.asarray(out)
            return out

        if not summarize:
            return format_out(outs)
    
        _statfuncs = [np.mean, ] if statfuncs is None else statfuncs

        eval_stats = lambda f,m: f(np.asarray([treeo[m] for treeo in outs]), axis=0)
        summary = [[eval_stats(f,m) for m, _ in enumerate(outs[0])] for f in _statfuncs] 

        return format_out(summary)
    return wrap

def nfiltercallwrapper(func):
    return lambda s, *a, **k: gscript(func)(*a, **(k | dict(self=s)))

class NodeFilterWrapper(): 
    def __init__(self, func = None):
        self.wrap = func
    
    @nfiltercallwrapper
    def __call__(gout, *args, **kwargs)->np.ndarray[bool]:
        return kwargs["self"].wrap(gout, *args, **kwargs)

    def __and__(self, other:NodeFilterWrapper | Callable | np.ndarray):
        if isinstance(other, Callable):
            return NodeFilterWrapper(lambda *a, **k: self(*a,**k) & other(*a, **k))
        if isinstance(other, np.ndarray):
            return NodeFilterWrapper(lambda *a, **k: self(*a,**k) & other)
        raise RuntimeError("Invalid __and__ operation") 
    
    def __or__(self, other:(NodeFilterWrapper | np.ndarray)):
        if isinstance(other, Callable):
            return NodeFilterWrapper(lambda *a, **k: self(*a,**k) | other(*a, **k))
        if isinstance(other, np.ndarray):
            return NodeFilterWrapper(lambda *a, **k: self(*a,**k) | other)
        raise RuntimeError("Invalid __or__ operation")
    
    def logical_not(self):
        return NodeFilterWrapper(lambda *a, **k: np.logical_not(self(*a,**k)))

    __invert__ = logical_not

    def freeze(self, **kwargs):
        return NodeFilterWrapper(lambda gout, *a, **k: self(gout, *a, **(k | kwargs)))

def freeze(func, **kwargs):
    return lambda gout, *a, **k: func(gout, *a, **(k | kwargs))

