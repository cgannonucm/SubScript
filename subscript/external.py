#!/usr/bin/env python3
from subscript.defaults import ParamKeys
import numpy as np

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
