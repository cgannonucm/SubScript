#!/usr/bin/env python
import h5py
from typing import Callable, Iterable
import numpy as np
import time
from collections import UserDict
from subscript.defaults import ParamKeys
from copy import copy
from subscript.defaults import Meta

class NodeProperties(UserDict):
    """
    Dictionary-like wrapper for Galacticus node data with filtering support.

    This class wraps a dictionary or UserDict containing node data from Galacticus outputs.
    It supports filtering nodes and slicing subsets of nodes for analysis.

    Parameters
    ----------
    d : dict or UserDict
        Dictionary-like object containing node data arrays or datasets.

    Raises
    ------
    RuntimeError
        If the input `d` is not a dictionary-like object.

    Attributes
    ----------
    _nodefilter : np.ndarray or None
        Boolean mask array for filtering nodes.
    _startn : int
        Start index for slicing nodes.
    _stopn : int or None
        Stop index for slicing nodes.

    Notes
    -----
    The class supports lazy loading from h5py datasets and callable objects that
    return arrays. When a filter is set, returned arrays are masked accordingly.
    """
    _nodefilter = None
    _startn = 0
    _stopn = None

    def __init__(self, d):
        out = super(NodeProperties,self).__init__()
        self.data = d

        if not isinstance(d,(dict, UserDict)):
            raise RuntimeError(f"d must be a dictionary! Not type {type(d)}")

        if isinstance(d, NodeProperties):
            self._startn = d._startn
            self._stopn =  d._stopn

    def __str__(self):
        return f"NodeData object"

    def __repr__(self):
        return f"NodeData object"

    def unfilter(self):
        # recursivly unfilter until reaching root
        if isinstance(self.data, UserDict):
            return self.data.unfilter()
        return NodeProperties(self) 

    def filter(self, nodefilter):
        out = NodeProperties(self)
        out._nodefilter = nodefilter
        return out

    def get_filter(self):
        if self._nodefilter is not None:
            return self._nodefilter
        return np.ones(self.data[next(self.data.__iter__())].shape[0], dtype=bool)

    def __getitem__(self, key): 
        # Allow for providing a set of keys
        if not isinstance(key, str):
            return [self[_key] for _key in key] 

        val = self.data[key]
        if isinstance(val, np.ndarray): 
            out = val
        elif isinstance(val, h5py.Dataset):
            _val = val[self._startn:self._stopn]
            if Meta.cache: 
                self.data[key] = _val
            out = _val
        elif isinstance(val, Callable):
            _val = val()[self._startn:self._stopn]
            if Meta.cache:
                self.data[key] = _val
            out = _val
        else:
            raise RuntimeError("Unrecognized Type") 

        if self._nodefilter is None:
            return out
        return out[self._nodefilter] 
                
def get_galacticus_outputs(galout:h5py.File)->np.ndarray[int]:
    """
    Extract sorted output snapshot indices from a Galacticus HDF5 file.

    Parameters
    ----------
    galout : h5py.File
        Open Galacticus HDF5 file containing simulation outputs.

    Returns
    -------
    np.ndarray[int]
        Sorted array of integer output snapshot indices extracted from the
        'Outputs' group keys.
    """
    output_groups:h5py.Group = galout["Outputs"] 

    outputs = np.zeros(len(output_groups), dtype=int)
    for n,key in enumerate(output_groups.keys()):
        outputs[n] = int(key[6:])
    return np.sort(outputs)

def get_custom_dsets(goutn:h5py.Group):
    """
    Generate example custom datasets from a Galacticus output group.

    Parameters
    ----------
    goutn : h5py.Group
        HDF5 group corresponding to a specific output snapshot in Galacticus data.

    Returns
    -------
    dict[str, Callable]
        Dictionary mapping custom dataset names to functions that return
        numpy arrays representing the dataset.

    Notes
    -----
    Example datasets include:
    - 'custom_node_tree': array indicating tree index for each node.
    - 'custom_node_tree_outputorder': array indicating output order index per node.
    - 'custom_id': unique node identifiers as an integer range.
    """
    """Example of custom datasets"""    

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
    """
    Load and tabulate node properties for all trees in a Galacticus output snapshot.

    Parameters
    ----------
    gout : h5py.File
        Dictionary-like structure containing Galacticus output node data.
    out_index : int, optional
        Index of the output snapshot to load (default is -1, which selects the
        latest available output).
    custom_dsets : Callable, optional
        Function that returns a dictionary of custom datasets for the output group.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    list[NodeProperties]
        List of NodeProperties objects, each corresponding to a single merger tree
        in the specified output snapshot.

    Notes
    -----
    Each NodeProperties instance contains node data arrays sliced for that tree,
    facilitating independent analysis of each tree's nodes.
    """
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

    get_cdsets = lambda i: {} if custom_dsets is None else custom_dsets
 
    props = get_cdsets(outn) | nodedata
    
    counts  = outn["mergerTreeCount"]

    out = []
    start = np.insert(np.cumsum(counts)[:-1], 0, 0)
    stop = np.cumsum(counts) 

    def gen_nodeproperties(n0, n1):
        #Carefull! Need copy here to avoid devious bugs             
        out = NodeProperties(copy(props))
        out._startn =  n0
        out._stopn  =  n1
        return out
        
    return [gen_nodeproperties(n0, n1) for n0, n1 in zip(start, stop)]
