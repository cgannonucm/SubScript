#!/usr/bin/env python3
'''Regression tests for subscript.tracking.track_subhalos.

Guards against the cross-tree nodeIndex splice: track_subhalos must follow a
subhalo by its PHYSICAL merger tree, not by a positional block index. Galacticus
may permute the per-output tree block order, and nodeIndex is only unique within
a tree (indexShift keeps an index stable across outputs but the same integer is
reused across trees). A positional-index tracker therefore follows different
physical trees across outputs and splices unrelated halos that share a nodeIndex.

These tests build a minimal synthetic Galacticus file in a tmp dir (so they do
not depend on the large real fixture) with:
  - two physical trees (ids 1 and 2),
  - a satellite nodeIndex 100 present in BOTH trees with a DISTINCT R_max
    (10.0 in tree 1, 999.0 in tree 2),
  - a per-output tree block order that PERMUTES (so positional index 0 is
    tree 1 at some outputs and tree 2 at others).

Tracking node 100 of physical tree 1 must return R_max == 10.0 at every output;
a positional tracker would splice in the 999.0 from tree 2 at the permuted
output.
'''
import numpy as np
import h5py
import pytest

from subscript.tracking import track_subhalos, _tree_position
from subscript.defaults import ParamKeys

RKEY = ParamKeys.dark_matter_profile_dmo_radius_velocity_max
# track_subhalos indexes nodedata output as nd[key_i][node_n]; nodedata collapses
# a single-key list to 1-D, so use >= 2 keys (as every real caller does).
PARAM_KEYS = [RKEY, ParamKeys.mass_bound]
HOST_RMAX = 50.0
SAT_INDEX = 100
SAT_RMAX = {1: 10.0, 2: 999.0}   # per physical tree


def _write_output(outputs_grp, out_num, tree_order, z):
    '''Write one Output{out_num} group. tree_order is the physical tree id per
    positional block; each tree contributes [host, satellite]. Node arrays are
    concatenated in block order, matching Galacticus' layout.'''
    idx, iso, bound, rmax, zli = [], [], [], [], []
    for tid in tree_order:
        # host (isolated)
        idx.append(tid); iso.append(1); bound.append(0.0)
        rmax.append(HOST_RMAX); zli.append(z)
        # satellite, shared nodeIndex across trees, tree-distinct R_max
        idx.append(SAT_INDEX); iso.append(0); bound.append(1.0e9)
        rmax.append(SAT_RMAX[tid]); zli.append(z)

    g = outputs_grp.create_group(f'Output{out_num}')
    g.create_dataset('mergerTreeIndex', data=np.asarray(tree_order, dtype=np.int64))
    g.create_dataset('mergerTreeCount', data=np.full(len(tree_order), 2, dtype=np.int64))
    nd = g.create_group('nodeData')
    nd.create_dataset('nodeIndex', data=np.asarray(idx, dtype=np.int64))
    nd.create_dataset(ParamKeys.is_isolated, data=np.asarray(iso, dtype=np.int64))
    nd.create_dataset(ParamKeys.mass_bound, data=np.asarray(bound, dtype=np.float64))
    nd.create_dataset(RKEY, data=np.asarray(rmax, dtype=np.float64))
    nd.create_dataset(ParamKeys.z_lastisolated, data=np.asarray(zli, dtype=np.float64))


def _make_file(path, orders):
    '''orders: dict {output_number: [physical tree ids in block order]}.
    Output numbers increase with cosmic time (higher = later = lower z).'''
    with h5py.File(path, 'w') as f:
        outs = f.create_group('Outputs')
        n = len(orders)
        for k, (out_num, order) in enumerate(sorted(orders.items())):
            _write_output(outs, out_num, order, z=float(n - 1 - k))


@pytest.fixture
def permuted_file(tmp_path):
    '''Latest output (3) has order [1,2]; the middle output (2) is PERMUTED to
    [2,1]; the earliest (1) is [1,2] again.'''
    p = tmp_path / 'permuted.hdf5'
    _make_file(p, {1: [1, 2], 2: [2, 1], 3: [1, 2]})
    return p


def test_tree_position_follows_physical_tree(permuted_file):
    with h5py.File(permuted_file, 'r') as f:
        # latest output 3: order [1,2]
        assert _tree_position(f, 3, 1) == 0
        assert _tree_position(f, 3, 2) == 1
        # permuted output 2: order [2,1] -> physical tree 1 is now block 1
        assert _tree_position(f, 2, 1) == 1
        assert _tree_position(f, 2, 2) == 0
        # absent tree -> None
        assert _tree_position(f, 2, 99) is None


def test_no_cross_tree_splice_permuted_order(permuted_file):
    '''Track node 100 of physical tree 1 (positional 0 at the latest output).
    Every recorded R_max must be tree 1's 10.0 -- never tree 2's 999.0.'''
    with h5py.File(permuted_file, 'r') as f:
        data, zsnaps = track_subhalos(f, nodeIndices=np.array([SAT_INDEX]),
                                      treeIndex=0, param_keys=PARAM_KEYS)
    rmax = data[SAT_INDEX][RKEY]
    assert rmax.shape == (3,)
    assert np.all(rmax == SAT_RMAX[1]), rmax
    assert not np.any(rmax == SAT_RMAX[2]), 'tree-2 value spliced in (cross-tree splice)'


def test_tracks_correct_tree_when_selected(permuted_file):
    '''Selecting the other physical tree (positional 1 at the latest output =
    tree 2) must return tree 2's values, confirming per-tree selection is right
    in both directions.'''
    with h5py.File(permuted_file, 'r') as f:
        data, _ = track_subhalos(f, nodeIndices=np.array([SAT_INDEX]),
                                 treeIndex=1, param_keys=PARAM_KEYS)
    rmax = data[SAT_INDEX][RKEY]
    assert np.all(rmax == SAT_RMAX[2]), rmax


def test_stable_order_is_a_noop(tmp_path):
    '''With a non-permuting order, physical-tree selection reduces to the plain
    positional selection: tree 1 -> 10.0, tree 2 -> 999.0 at every output.'''
    p = tmp_path / 'stable.hdf5'
    _make_file(p, {1: [1, 2], 2: [1, 2], 3: [1, 2]})
    with h5py.File(p, 'r') as f:
        d0, _ = track_subhalos(f, np.array([SAT_INDEX]), treeIndex=0, param_keys=PARAM_KEYS)
        d1, _ = track_subhalos(f, np.array([SAT_INDEX]), treeIndex=1, param_keys=PARAM_KEYS)
    assert np.all(d0[SAT_INDEX][RKEY] == SAT_RMAX[1])
    assert np.all(d1[SAT_INDEX][RKEY] == SAT_RMAX[2])
