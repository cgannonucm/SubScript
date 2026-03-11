"""
Subhalo time-series tracking across Galacticus output snapshots.

This module provides functions to extract the evolution history of individual
subhalos over cosmic time by reading every output snapshot stored in a
Galacticus HDF5 file.

.. note::
   To use :func:`track_subhalos`, Galacticus must be run with the
   ``nodeOperator`` set to ``indexShift`` so that node indices remain
   consistent across snapshots::

       <nodeOperator value="indexShift" />

Functions
---------
:func:`track_subhalos`
    Extract time-series data for a set of subhalo node indices across all
    snapshots of a single merger tree.

:func:`track_subhalo`
    Filter the per-subhalo time-series returned by :func:`track_subhalos` to
    retain only the snapshots where the subhalo is a bound satellite (not
    isolated and with positive bound mass).
"""
import numpy as np
import subscript.scripts.nfilters as nf
from subscript.tabulatehdf5 import get_galacticus_outputs, tabulate_trees
from subscript.scripts.nodes import nodedata
from subscript.defaults import ParamKeys

def track_subhalos(galacticus_out, nodeIndices, treeIndex,  param_keys = None):
    """Extract time-series data for specified subhalo nodes across all Galacticus snapshots.
    NOTE: To use this function, galacticus must be run with the nodeOperator indexShift.
    Ie  
    ```
    <nodeOperator value="indexShift" />
    ```

    
    Parameters
    ----------
    galacticus_out : h5py.File
        Opened HDF5 file object containing Galacticus simulation output
    nodeIndices : array-like
        Array of node indices for subhalos to track
    treeIndex : int
        Index of the merger tree to extract data from
    param_keys : list of str, optional
        List of parameter keys to extract for each subhalo. If None, extracts all available keys
        from the tree at the first snapshot
    
    Returns
    -------
    subhalo_data : dict
        Nested dictionary with structure {node_id: {param_key: time_series_array, 'zsnap': redshift_array}}
    zsnaps : ndarray
        Array of redshifts at each snapshot (averaged over host halos)
    """
    snaps = np.flip(np.asarray(get_galacticus_outputs(galacticus_out)))

    param_keys = param_keys if param_keys is not None else [_key for _key in tabulate_trees(galacticus_out, snaps[0])[treeIndex].keys()]

    subhalo_data = {id : {key: np.zeros(len(snaps)) for key in param_keys} | {'zsnap' : np.zeros(len(snaps))} for id in nodeIndices}

    zsnaps = np.zeros(len(snaps))

    for j, isnap in enumerate(snaps):
        snap = tabulate_trees(galacticus_out, isnap)[treeIndex]  
        nd = nodedata(snap, key=param_keys)    

        ids = nodedata(snap, 'nodeIndex')

        zsnaps[j] = np.mean(nodedata(snap, ParamKeys.z_lastisolated, nfilter=nf.hosthalos))

        for n, id in enumerate(ids):
            if id not in nodeIndices:
                continue 
            for i, key in enumerate(param_keys):
                subhalo_data[id][key][j] = nd[i][n]
                        
    return subhalo_data, zsnaps
 
def track_subhalo(subhalos_over_time, zsnaps, nodeindex, param_keys):
    """Filter subhalo time-series data to retain only non-isolated, bound snapshots. 
    Parameters
    ----------
    subhalos_over_time : dict
        Nested dictionary output from track_subhalos containing time-series for each subhalo
    zsnaps : ndarray
        Array of redshifts at each snapshot
    nodeindex : int
        Node index of the subhalo to filter
    param_keys : list of str
        List of parameter keys to include in filtered output
    
    Returns
    -------
    filtered_data : dict
        Dictionary with structure {param_key: filtered_time_series_array} where filtering removes
        snapshots where the subhalo is isolated (is_isolated == 1) or has no bound mass (mass_bound <= 0)
    filtered_zsnaps : ndarray
        Filtered array of redshifts corresponding to non-isolated, bound snapshots
    """
    _filter = (subhalos_over_time[nodeindex][ParamKeys.is_isolated] == 0) & (subhalos_over_time[nodeindex][ParamKeys.mass_bound] > 0)
    return {key: subhalos_over_time[nodeindex][key][_filter] for key in param_keys}, zsnaps[_filter]