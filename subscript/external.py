#!/usr/bin/env python3
"""
Adapters for converting external simulation data into Galacticus-compatible formats.

Currently supports the **Symphony** suite of zoom-in simulations
(Nadler et al. 2023; https://arxiv.org/abs/2206.00038).  The main entry point
is :func:`symphony_to_galacticus_like_dict`, which converts a Symphony
per-snapshot subhalo catalogue into a flat dictionary whose keys match
:class:`~subscript.defaults.ParamKeys` string constants — the same format
produced by :func:`~subscript.tabulatehdf5.tabulate_trees`.

The mapping between Symphony catalogue columns and Galacticus-style keys is
controlled by :data:`KEY_MAP_SYMPHONY_DEFAULT`.
"""
from subscript.defaults import ParamKeys
import numpy as np

#: Default column mapping used by :func:`symphony_to_galacticus_like_dict`.
#:
#: This is a ``dict`` of ``dict`` with the following structure::
#:
#:     {
#:         <ParamKeys attribute (str)>: {
#:             'SymphonyName': <Symphony catalogue column name (str)>,
#:             'PerSnap':      <True if the column varies per snapshot,
#:                              False if it is a halo-history quantity>,
#:             'conversion':   <multiplicative factor applied to convert the
#:                              Symphony value to the Galacticus unit convention>,
#:         },
#:         ...
#:     }
#:
#: **PerSnap flag**
#:
#: * ``True``  – the value is read from the per-snapshot subhalo array
#:   (``sim_data[0]``) at column ``isnap``.
#: * ``False`` – the value is read from the halo-history array
#:   (``sim_data[1]``), which has one entry per subhalo regardless of snapshot.
#:
#: **Conversion factors**
#:
#: Symphony masses are in *M☉* and distances in *kpc/h*.  The conversion
#: factors below bring values into Galacticus conventions (masses in *M☉*,
#: distances in *Mpc*):
#:
#: * Masses: ``1.0`` (Symphony M☉ ↔ Galacticus M☉, no change).
#: * Distances/radii: ``1e-3`` (Symphony kpc → Galacticus Mpc).
KEY_MAP_SYMPHONY_DEFAULT = {
                        ParamKeys.mass_basic    : {
                                                   'SymphonyName' : 'mpeak',
                                                   'PerSnap'      : False,
                                                   'conversion'   : 1.0,
                                                  },
                        ParamKeys.mass_bound    : {
                                                   'SymphonyName' : 'mvir',
                                                   'PerSnap'      : True,
                                                   'conversion'   : 1.0,
                                                  },
                        ParamKeys.rvir          : {
                                                   'SymphonyName' : 'rvir',
                                                   'PerSnap'      : True,
                                                   'conversion'   : 1E-3,
                                                  },
                        ParamKeys.x             : {
                                                   'SymphonyName' : 'x',
                                                   'PerSnap'      : True,
                                                   'conversion'   : 1E-3,
                                                  },
                        ParamKeys.y             : {
                                                   'SymphonyName' : 'x',
                                                   'PerSnap'      : True,
                                                   'conversion'   : 1E-3,
                                                  },
                        ParamKeys.z             : {
                                                   'SymphonyName' : 'x',
                                                   'PerSnap'      : True,
                                                   'conversion'   : 1E-3,
                                                  },
                        ParamKeys.z_lastisolated: {
                                                   'SymphonyName' : 'merger_snap',
                                                   'PerSnap'      : False,
                                                   'conversion'   : 1.0,
                                                  },
                        'custom_id'             : {
                                                   'SymphonyName' : 'id',
                                                   'PerSnap'      : True,
                                                   'conversion'   : 1.0,
                                                  },
                        }

def symphony_to_galacticus_like_dict(sim_data, z_snap, key_map=KEY_MAP_SYMPHONY_DEFAULT, isnap=-1, tree_index=1):
    """
    Convert a Symphony simulation subhalo catalogue to a Galacticus-compatible dictionary.

    Reads subhalo properties from a Symphony snapshot and re-maps them to the
    key convention used by Galacticus (see :class:`~subscript.defaults.ParamKeys`).
    The resulting dictionary can be passed directly to any SubScript analysis
    function that accepts Galacticus-like node data.

    Parameters
    ----------
    sim_data : tuple of array-like
        Two-element tuple ``(halos, halo_history)`` as returned by
        ``symphony.load()``:

        * ``halos`` – structured per-snapshot array with shape ``(N, n_snaps)``.
        * ``halo_history`` – structured history array with shape ``(N,)``.

        An ``ok`` boolean column in ``halos`` is used to mask subhalos that
        are not present at the requested snapshot.
    z_snap : array-like of float
        Redshift values for every Symphony snapshot, indexed by snapshot number.
        Used to convert the integer ``merger_snap`` column to a redshift value.
    key_map : dict, optional
        Mapping that controls which Symphony columns are extracted and how they
        are renamed and scaled.  Defaults to :data:`KEY_MAP_SYMPHONY_DEFAULT`.
        See the :data:`KEY_MAP_SYMPHONY_DEFAULT` documentation for the expected
        structure.
    isnap : int, optional
        Snapshot index to extract per-snapshot quantities from.  Defaults to
        ``-1`` (the last/most-recent snapshot).
    tree_index : int, optional
        Integer label assigned to every subhalo in the ``custom_node_tree``
        output column.  Defaults to ``1``.

    Returns
    -------
    dict
        Flat dictionary whose keys are :class:`~subscript.defaults.ParamKeys`
        string values (plus ``'custom_node_tree'`` and
        ``ParamKeys.is_isolated``).  Values are 1-D :class:`numpy.ndarray`
        objects with one entry per subhalo.

        Special entries always present in the output:

        * ``'custom_node_tree'`` – integer array filled with ``tree_index``.
        * :attr:`~subscript.defaults.ParamKeys.is_isolated` – ``0`` for all
          subhalos except the first (index 0), which is treated as the host
          halo and set to ``1``.
    """
    ok = sim_data[0]['ok'][:, isnap]
    h, hist = sim_data[0][ok], sim_data[1][ok]
    out = {}

    for gparamkey, symmap in key_map.items():
        if symmap['PerSnap']:
            val = np.astype(h[symmap['SymphonyName']][:, isnap], float)
        else:
            val = np.astype(hist[symmap['SymphonyName']], float)

        val *= symmap['conversion']

        # Hard code special cases
        coord_indexes = {
                         ParamKeys.x: 0,
                         ParamKeys.y: 1,
                         ParamKeys.z: 2
                        }

        # Split coordinates from 3 vector into individual entries
        if gparamkey in coord_indexes.keys():
            n = coord_indexes[gparamkey]
            val = val[:, n]

        # Only the snapshot indexes are stored, get the redshift for the given snapshot index
        if gparamkey == ParamKeys.z_lastisolated:
            val = z_snap[np.astype(val, int)]

        out[gparamkey] = val

    # Assign tree index to all subhalos
    nodecount = out.values().__iter__().__next__().shape[0]
    out['custom_node_tree'] = tree_index * np.ones(nodecount, dtype=int)

    # The first halo is the host
    out[ParamKeys.is_isolated] = np.zeros(nodecount, dtype=int)
    out[ParamKeys.is_isolated][0] = 1

    return out
