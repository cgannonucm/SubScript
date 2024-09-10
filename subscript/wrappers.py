#!/usr/bin/env python
from __future__ import annotations
from typing import Any, Callable, Iterable, ParamSpec, Concatenate, TypeVar, Generic
from functools import wraps
import numpy as np
import h5py

from subscript.tabulatehdf5 import nodeProperties, tabulate_nodes
from subscript import defaults

def nodedata(gout, out_index=-1):
    if isinstance(gout, nodeProperties):
        _gout = gout
    elif isinstance(gout, h5py.File):
        # TODO: Refactor key_index to tree_inxex
        _gout = tabulate_nodes(gout, key_index=out_index)
    elif isinstance(gout, dict):
        _gout = nodeProperties(gout)
    else:
        raise RuntimeError("Unrecognized data type for gout")
    return _gout

def nfiltercallwrapper(func):
    def wrap(self, gout:(h5py.File | nodeProperties | dict), *args, out_index=-1, plabels = None, **kwargs):
        plabels = {} if plabels is None else plabels
        return func(self, nodedata(gout, out_index=out_index), 
                        *args, out_index=-1, 
                        nplabels=(defaults.plabels | plabels),**kwargs)
    return wrap

class NodeFilterWrapper(): 
    def __init__(self, func = None):
        self.wrap = func
    
    @nfiltercallwrapper
    def __call__(gout, *args, **kwargs)->np.ndarray[bool]:
        return self.wrap(*args, **kwargs)

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

    def freeze(self, *args, **kwargs):
        return lambda gout: self(gout, *args, **kwargs)

# This design is chosen to allow for lsps 
# basically impossible impossible to get 
# type hints unless we design our function like this
class _nfilter_all(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(self, gout, **kwargs):
        return np.ones(gout[next(iter(gout))].shape, dtype=bool)

nfilter_all = _nfilter_all()

class _nfilter_halos(NodeFilterWrapper):
    @nfiltercallwrapper
    def __call__(self, gout, **kwargs):
        pass

def gscript(func):
    def wrap(gout:(h5py.File | nodeProperties | dict), *args, 
                nodefilter=None, treestats=False, treestatfuncs:Iterable[Callable] = None,
                out_index=-1, tree_index = None, plabels = None,
                **kwargs): 
        plabels = {} if plabels is None else plabels
          
        _gout = nodedata(gout, out_index)

        trees = np.unique(_gout["custom_node_tree"])
        if tree_index is not None:
            trees = (tree_index)

        if nodefilter is None:
            nodefilter = np.ones(_gout[next(iter(_gout))].shape, dtype=bool)

        if isinstance(nodefilter, np.ndarray):
            _nodefilter = nodefilter
        elif isinstance(nodefilter, Callable):
            _nodefilter = nodefilter(_gout, *args, **kwargs)
        else:
            raise RuntimeError("Unrecognized data type for nodefilter")
          
        summary = None

        for treen, tree in enumerate(trees):
            filtertree = _gout["custom_node_tree"] == tree
            _gout_ftree    = _gout.filter(filtertree)
            _gout_ftree_nf = _gout.filter(filtertree & _nodefilter)
    
            out = func(_gout_ftree_nf, *args, 
                        nodefilter=_nodefilter, **kwargs, 
                        gout_ftree=_gout_ftree,
                        tree_index=tree,
                        nplabels=(defaults.plabels | plabels))                    

            single_out = isinstance(out, np.ndarray)

            _out = out
            if single_out:
                _out = [out, ]
            
            if summary is None:
                summary = []
                for o in _out:
                    summary.append([[] for tree in trees])

            for n,o in enumerate(_out):
                summary[n][treen] = o 
 
        format_out = lambda o: o[0] if single_out else o

        if not treestats:
            return format_out(summary)

        if tree_index is not None:
            return format_out([out[0] for out in summary])

        _treestatfuncs = (np.mean, ) if treestatfuncs is None else treestatfuncs        

        stat_summary = []
        for _func in _treestatfuncs:
            stat_summary.append([])
            for arr in summary:
                stat_summary[-1].append(_func(arr, axis=0))

        if single_out:
            return stat_summary[0]
        return stat_summary 
    return wrap

@gscript
def nodevalue(gout, label:(str | Iterable[str]), **kwargs):
    return gout[label]

def main():
    path_dmo = "../data/test.hdf5"
    gout = h5py.File(path_dmo)
    print(len(nodevalue(gout, defaults.plabels["mass"])))

if __name__ == "__main__": 
    main()