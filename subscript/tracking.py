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

def _tree_order(galacticus_out, isnap):
    """Physical ``mergerTreeIndex`` id sitting at each positional tree block of output ``isnap``.

    Galacticus writes the per-output tree blocks in an order that can permute
    from output to output, so a positional block index does NOT correspond to a
    fixed physical tree across snapshots. This returns the physical tree id at
    each block position for a single output, so callers can resolve a positional
    index to a snapshot-stable physical id.
    """
    return np.asarray(galacticus_out["Outputs"][f"Output{int(isnap)}"]["mergerTreeIndex"][:])

def _tree_position(galacticus_out, isnap, phys_tree):
    """Positional block index of physical tree ``phys_tree`` at output ``isnap``.

    Returns ``None`` if that physical tree has no block at this output.
    """
    pos = np.flatnonzero(_tree_order(galacticus_out, isnap) == phys_tree)
    return int(pos[0]) if pos.size else None

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
        Positional index of the merger tree in the latest output's tree list.
        This is resolved once to a physical ``mergerTreeIndex`` id, which is
        stable across snapshots, and that physical id is used to select the tree
        block at every output (see Notes).
    param_keys : list of str, optional
        List of parameter keys to extract for each subhalo. If None, extracts all available keys
        from the tree at the first snapshot
    
    Returns
    -------
    subhalo_data : dict
        Nested dictionary with structure {node_id: {param_key: time_series_array, 'zsnap': redshift_array}}
    zsnaps : ndarray
        Array of redshifts at each snapshot (averaged over host halos)

    Notes
    -----
    Nodes are followed by physical tree id, NOT by positional block index.
    Galacticus can permute the per-output tree block order, and ``nodeIndex`` is
    only unique *within* a tree (``indexShift`` keeps an index stable across
    outputs but the same integer is reused across trees). Selecting a fixed
    positional block index therefore follows different physical trees across
    outputs and, via the shared ``nodeIndex``, splices unrelated halos into one
    time series. Resolving ``treeIndex`` to a physical ``mergerTreeIndex`` id and
    selecting by that id at every output removes the splice.
    """
    snaps = np.flip(np.asarray(get_galacticus_outputs(galacticus_out)))

    # Resolve the caller's positional treeIndex (into the latest output's tree
    # list) to a physical mergerTreeIndex id, then select by that id at every
    # output so a permuting block order can't swap in a different physical tree.
    phys_tree = int(_tree_order(galacticus_out, snaps[0])[treeIndex])

    param_keys = param_keys if param_keys is not None else [_key for _key in tabulate_trees(galacticus_out, snaps[0])[treeIndex].keys()]

    subhalo_data = {id : {key: np.zeros(len(snaps)) for key in param_keys} | {'zsnap' : np.zeros(len(snaps))} for id in nodeIndices}

    zsnaps = np.zeros(len(snaps))

    for j, isnap in enumerate(snaps):
        pos = _tree_position(galacticus_out, isnap, phys_tree)
        if pos is None:
            continue  # physical tree absent at this output; leave zeros (filtered downstream)

        snap = tabulate_trees(galacticus_out, isnap)[pos]
        nd = nodedata(snap, key=param_keys)

        ids = nodedata(snap, 'nodeIndex')

        zsnaps[j] = np.mean(nodedata(snap, ParamKeys.z_lastisolated, nfilter=nf.hosthalos))

        for n, id in enumerate(ids):
            if id not in nodeIndices:
                continue
            for i, key in enumerate(param_keys):
                subhalo_data[id][key][j] = nd[i][n]

    return subhalo_data, zsnaps
 
def track_subhalo(subhalos_over_time, zsnaps, nodeindex, param_keys, include_isolated=False):
    """Filter subhalo time-series data to retain only bound snapshots.
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
    include_isolated : bool, optional
        If True, include all snapshots regardless of isolation status. Only
        unbound mass is still filtered out. Default is False.

    Returns
    -------
    filtered_data : dict
        Dictionary with structure {param_key: filtered_time_series_array} where filtering removes
        snapshots where the subhalo has no bound mass (mass_bound <= 0), and optionally where
        the subhalo is isolated (is_isolated == 1) when include_isolated is False.
    filtered_zsnaps : ndarray
        Filtered array of redshifts corresponding to retained snapshots
    """
    _filter = subhalos_over_time[nodeindex][ParamKeys.mass_bound] > 0
    if not include_isolated:
        _filter = _filter & (subhalos_over_time[nodeindex][ParamKeys.is_isolated] == 0)
    return {key: subhalos_over_time[nodeindex][key][_filter] for key in param_keys}, zsnaps[_filter]