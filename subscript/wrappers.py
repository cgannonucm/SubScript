#!/usr/bin/env python
from __future__ import annotations
from typing import Any, Callable, Iterable, List
from collections import UserDict
from functools import reduce
import numpy as np
import h5py

from subscript.tabulatehdf5 import NodeProperties, tabulate_trees
from subscript import tabulatehdf5
from subscript import defaults

def reduce_input(l, out=None):
    if out is None:
        out = []
    
    for i in l:
        if isinstance(i, (dict, UserDict)):
            out.append(i)
            continue
        reduce_input(i, out)
    return out
 
def format_nodedata(gout, out_index=-1)->Iterable[NodeProperties]:
    if isinstance(gout, (dict, UserDict)):
        _gout = [NodeProperties(gout), ]
    elif isinstance(gout, h5py.File):
        _gout = tabulate_trees(gout, out_index=out_index)
    elif isinstance(gout, Iterable): 
        _gout = reduce_input([format_nodedata(o, out_index=out_index) for o in gout])
    else:
        raise RuntimeError(f"Unrecognized data type for gout {type(gout)}")
    return _gout

def gscript(func):
    def wrap(gout:(h5py.File | NodeProperties | dict), 
                *args, 
                nfilter:(Callable | np.ndarray[bool])=None, 
                summarize:bool=False, 
                statfuncs:Iterable[Callable] = None,
                out_index:int=-1,
                **kwargs):         
        if gout is None:
            print(kwargs)
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
        ntrees = len(trees)

        for nodestree in trees:
            _nodestree = nodestree.unfilter()
            _nodefilter = None
            if isinstance(nfilter, Callable):
                _nodefilter = nfilter(_nodestree, **kwargs)
            elif isinstance(nfilter,np.ndarray):
                _nodefilter = nfilter
            _nodestree_filtered = _nodestree.filter(_nodefilter)
            o = func(_nodestree_filtered, *args, **(kwargs | dict(nfilter=_nodefilter)))
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

        if isinstance(outs[0], Iterable):
            eval_stats = lambda f,m: f(np.asarray([treeo[m] for treeo in outs]), axis=0)
            summary = [[eval_stats(f,m) for m, _ in enumerate(outs[0])] for f in _statfuncs] 
        else:
            eval_stats = lambda f: f(np.asarray([treeo for treeo in outs]), axis=0)
            summary = [eval_stats(f) for f in _statfuncs] 

        return format_out(summary)
    return wrap

def gscript_proj(func):
    """
    Wraper for scripts that involve projection, allows  passing of multiple normal vectors.
    If multiple projection vectors are passed, they are treated as seperate "trees".
    """
    def wrap(gout, normvector, *args, **kwargs):
        n = None

        @gscript
        def wrap_inner(gout, *args, normvector, **kwargs):
            nonlocal n
            v =  normvector
            if n is not None:
                v = normvector[n]
                n += 1  
                if n >= len(normvector):
                    n = 0


            return func(gout, *args, normvector=v, **kwargs)


        if isinstance(normvector, np.ndarray) and normvector.ndim == 1: 
            return wrap_inner(gout, *args, normvector=normvector, **kwargs)

        n = 0
        return wrap_inner([gout for _ in normvector], *args, normvector=normvector, **kwargs)

    return wrap

def freeze(func, **kwargs):
    return lambda gout, *a, **k: func(gout, *a, **(k | kwargs))

def multiproj(func, nfilter):
    return gscript_proj(freeze(func, nfilter=nfilter))


