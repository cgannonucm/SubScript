#!/usr/bin/env python3
"""
High-level helper for extracting and caching per-subhalo time-series data.

This module provides :func:`subhalo_timeseries`, a convenience wrapper around
:func:`~subscript.tracking.track_subhalos` and
:func:`~subscript.tracking.track_subhalo` that:

* Retrieves all subhalo node IDs at the last snapshot of a given merger tree.
* Tracks those subhalos across every available Galacticus snapshot.
* Filters each subhalo's history to satellite-only epochs (removing snapshots
  where the halo is isolated or has zero bound mass).
* Caches results to a ``.pkl`` file alongside the input HDF5 file so that
  repeated calls are fast.

.. note::
   Galacticus must be run with ``<nodeOperator value="indexShift" />`` for
   node indices to remain stable across snapshots (required by the underlying
   :func:`~subscript.tracking.track_subhalos` function).
"""
import os
import pickle
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import h5py
import numpy as np

from subscript.tabulatehdf5 import tabulate_trees
from subscript.scripts.nodes import nodedata
from subscript.scripts import nfilters as nf
from subscript.tracking import track_subhalos, track_subhalo


def _subscript_version():
    """Installed subscript (subhaloscript) version string used to stamp caches.

    Reflects the *installed* package metadata, which is the correct notion of
    "the subscript that produced a cache". For an editable install this is
    frozen at install time and only refreshes on reinstall. Returns 'unknown'
    if the package is not installed via metadata, in which case it never matches
    a stamped version and caches are always regenerated (safe).
    """
    try:
        return version('subhaloscript')
    except PackageNotFoundError:
        return 'unknown'


def subhalo_timeseries(galacticus_hdf5: h5py.File, tree_index: int, refresh=False, include_isolated=False) -> dict:
    """
    Extract per-subhalo time-series data for all subhalos in a Galacticus tree.

    Retrieves all subhalo node IDs at the last snapshot of the given tree, runs
    track_subhalos across all snapshots, then filters each subhalo's time-series
    via track_subhalo (removing isolated unless include_isolated is True). Results
    are cached to disk using pickle. Each cache is stamped with the subscript
    version that produced it and is transparently regenerated when the installed
    subscript version changes (or when an older, unversioned cache is found).

    Parameters
    ----------
    galacticus_hdf5 : h5py.File
        Open HDF5 file object for the Galacticus simulation output.
    tree_index : int
        Index of the merger tree to process.
    refresh : bool, optional
        If True, bypass cache and recompute results. Default is False.
    include_isolated : bool, optional
        If True, include all snapshots regardless of isolation status (only
        unbound mass is still filtered out). Default is False.

    Returns
    -------
    dict
        Dictionary mapping node IDs (int) to dicts with keys:
        - 'data': dict of {param_key: time_series_array}
        - 'zsnaps': corresponding redshift array
    """
    file_path = galacticus_hdf5.filename

    # Build cache filename: {stem}-{hash[:16]}-tree{tree_index}[-isolated].pkl
    file_stem = Path(file_path).stem
    file_hash = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()[:16]
    isolated_tag = "-isolated" if include_isolated else ""
    cache_path_name = f"{file_stem}-{file_hash}-tree{tree_index}{isolated_tag}.pkl"
    cache_path = os.path.join(Path(file_path).parent, cache_path_name)

    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        # Reuse only if the cache was produced by the current subscript version.
        # Older, unversioned caches are raw dicts keyed by int node_id, so the
        # string-key lookup returns None (no collision) -> treated as stale.
        if isinstance(cached, dict) and cached.get('subscript_version') == _subscript_version():
            return cached['result']
        # stale (version mismatch or old unversioned format): fall through,
        # recompute, and overwrite cache_path below.

    # Get all subhalo node IDs at the last output of this tree
    trees = tabulate_trees(galacticus_hdf5)
    node_ids = nodedata(trees[tree_index], key='nodeIndex', nfilter=nf.subhalos)

    # Track all subhalos across all snapshots
    dat_subhalos, zsnap_subhalos = track_subhalos(
        galacticus_hdf5,
        nodeIndices=node_ids,
        treeIndex=tree_index
    )

    # Determine which param_keys were tracked (exclude the internal 'zsnap' key)
    first_id = node_ids[0]
    param_keys = [k for k in dat_subhalos[first_id].keys() if k != 'zsnap']

    # Build result dict: filter each subhalo's time-series to satellite-only snapshots
    result = {}
    for node_id in node_ids:
        filtered_data, filtered_zsnaps = track_subhalo(
            dat_subhalos, zsnap_subhalos, node_id, param_keys,
            include_isolated=include_isolated
        )
        result[node_id] = {'data': filtered_data, 'zsnaps': filtered_zsnaps}

    with open(cache_path, 'wb') as f:
        pickle.dump({'subscript_version': _subscript_version(), 'result': result}, f)

    return result
