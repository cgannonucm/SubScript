#!/usr/bin/env python
import h5py
from typing import Callable, Iterable
import numpy as np
import time
from collections import UserDict
from subscript.defaults import ParamKeys, Meta
from functools import cache
from copy import copy


# Hacky workaround to get caching working
@cache
def _get_custom_cache(h, key):
    _hash = hash(str(h) + key)
    return NodeProperties._cache.get(_hash)
    
def _set_custom_cache(h, key, val):
    _hash = hash(str(h) + key)
    NodeProperties._cache[_hash] = val
    
class NodeProperties(UserDict):
    _counter = 0
    _cache = {}

    _nodefilter = None
    _startn = 0
    _stopn = None
    _fname = None

    def __init__(self, d, nodefilter:np.ndarray=None, startn = None, stopn=None, fpath=None):
        self._nodefilter = nodefilter 
        out = super(NodeProperties,self).__init__()
        self.data = d
        NodeProperties._counter += 1

        if not isinstance(d,(dict, UserDict)):
            raise RuntimeError(f"d must be a dictionary! Not type {type(d)}")

        if isinstance(d, NodeProperties):
            if startn is not None or stopn is not None:
                raise RuntimeError("Changing indexes is unsorported.")
            self._startn = d._startn
            self._stopn = d._stopn 
            self._fpath = d._fpath
        else:
            self._startn = 0 if startn is None else startn
            self._stopn = None if stopn is None else stopn
            self._fpath = fpath

        hashstr = str(self._fpath) + f"-from({self._startn}:{self._stopn})"
        hashstr += f"-id-{self._counter}" if self._fpath is None else ""
        hashstr += "-nofilter" if nodefilter is None else f"-filter-bytes-hash" + str(hash(nodefilter.data.tobytes()))

        self._data = self.data
   
        self._hashstr = hashstr

        self._hash = hash(hashstr)

    def __str__(self):
        return f"NodeData object"

    def __repr__(self):
        return f"NodeData object"

    def __hash__(self):
        return self._hash

    def unfilter(self):
        if self._nodefilter is not None:
            return NodeProperties(self.data.unfilter())
        return NodeProperties(self)

    def filter(self, nodefilter):
        return NodeProperties(self,nodefilter)

    def get_filter(self):
        return self._nodefilter

    def _cached(self, key):
        _c = _get_custom_cache(self._hash, key)
        if _c is not None:
            return _c 

        if self._nodefilter is not None:
            val = self.unfilter()[key]
        else:
            val = self.data[key]

        if isinstance(val, np.ndarray): 
            out = val
        elif isinstance(val, h5py.Dataset):
            out = val[self._startn:self._stopn]
        elif isinstance(val, Callable):
            out = val()[self._startn:self._stopn]
        else:
            raise RuntimeError("Unrecognized Type") 

        if self._nodefilter is None:
            _out = out
        else:
            _out = out[self._nodefilter]

        _set_custom_cache(self._hash, key, _out)

        return _out

    def _get_item(self, key):    
        if Meta.enable_higher_order_caching or self._nodefilter is None: 
            return self._cached(key)
    
        v = self.unfilter()[key]
        if self._nodefilter is not None:
            return v[self._nodefilter]
        return v
    
    def __getitem__(self, key): 
        # Allow for providing a set of keys
        if not isinstance(key, str):
            return [self[_key] for _key in key] 
        
        # Cache based on file name or fall back to id
        return self._get_item(key)
                
def get_galacticus_outputs(galout:h5py.File)->np.ndarray[int]:
    output_groups:h5py.Group = galout["Outputs"] 

    outputs = np.zeros(len(output_groups), dtype=int)
    for n,key in enumerate(output_groups.keys()):
        outputs[n] = int(key[6:])
    return np.sort(outputs)

def get_custom_dsets(goutn:h5py.Group):
    """Generates standar custom datasets to make data analysis easier"""    

    # Total number of nodes
    nodecount = np.sum(goutn["mergerTreeCount"][:]) 
    # Node counts
    counts    = goutn["mergerTreeCount"]
    # Tree indexes
    treenums  = goutn["mergerTreeIndex"]

    nodedata  = goutn["nodeData"]
    
    return {
        "custom_node_tree"            : lambda : np.concatenate([np.full(count,index) for (count,index) in zip(counts,treenums)]),
        "custom_node_tree_outputorder": lambda : np.concatenate([np.full(count,i) for (i,count) in enumerate(counts)]),
        "custom_id"                   : lambda : np.arange(nodecount),
    }


def tabulate_trees(gout:h5py.File, out_index:int=-1, custom_dsets:Callable = None, **kwargs)->NodeProperties:
    """Reads node propreties from a galacticus HDF5 file"""
    outs = gout["Outputs"] 

    _key_index = out_index
    if out_index == -1:
        _key_index = np.max(get_galacticus_outputs(gout))
    
    outn:h5py.Group = outs[f"Output{_key_index}"]
    nd:h5py.Group   = outn["nodeData"]

    # Total number of nodes in output can be obtained by summing this dataset
    # Which contains the number of nodes per tree
    nodecount = np.sum(outn["mergerTreeCount"][:]) 

    nodedata = {}
    for key, val in nd.items():
        # Add to return dictionary if we have a dateset that matches the total number of nodes
        if not isinstance(val, h5py.Dataset):
            continue
        if val.shape[0] != nodecount:
            continue
        # First entry in tuple marks this as a h5py dataset 
        nodedata[key] = val

    get_cdsets = get_custom_dsets if custom_dsets is None else custom_dsets
 
    props = get_cdsets(outn) | nodedata
    
    counts  = outn["mergerTreeCount"]

    out = []
    start = np.insert(np.cumsum(counts)[:-1], 0, 0)
    stop = np.cumsum(counts) 

    #Carefull! Need copy here to avoid devious bugs
    return [NodeProperties(copy(props), nodefilter=None, startn=n0, stopn=n1, fpath=gout.filename) for n0, n1 in zip(start, stop)]

def main():
    path_dmo = "../data/test.hdf5"
    gout = h5py.File(path_dmo)

    trees = tabulate_trees(gout)

    total_count = 0
    for tree in trees: 
        total_count += len(tree["basicMass"])

    assert(total_count == np.sum(gout["Outputs"]["Output1"]["mergerTreeCount"][:]))    

        

if __name__ == "__main__": 
    main()