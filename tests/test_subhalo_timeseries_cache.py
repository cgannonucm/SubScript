#!/usr/bin/env python3
'''Tests for the version-stamped subhalo_timeseries disk cache.

Each cache pickle is wrapped as {'subscript_version': <ver>, 'result': <dict>}
and is only reused when the stamped version matches the currently installed
subscript version; otherwise (version mismatch, or an older unversioned raw-dict
cache) it is transparently regenerated and overwritten. refresh=True always
recomputes.

Uses a minimal synthetic Galacticus file built in a tmp dir (same pattern as
tests/test_tracking.py) so the cache lands in tmp_path and no large fixture is
needed.
'''
import pickle

import numpy as np
import h5py
import pytest

from subscript.subhalo_timeseries import subhalo_timeseries, _subscript_version
from subscript.defaults import ParamKeys

RKEY = ParamKeys.dark_matter_profile_dmo_radius_velocity_max
SAT_INDEX = 100
SAT2_INDEX = 200
SAT_RMAX = {1: 10.0, 2: 999.0}
NODES_PER_TREE = 3   # host + 2 satellites (>1 subhalo so node_ids stays an array)


def _write_output(outputs_grp, out_num, tree_order, z):
    # per tree block: [host, satellite 100, satellite 200]
    idx, iso, bound, rmax, zli = [], [], [], [], []
    for tid in tree_order:
        idx += [tid, SAT_INDEX, SAT2_INDEX]; iso += [1, 0, 0]
        bound += [0.0, 1.0e9, 5.0e8]; rmax += [50.0, SAT_RMAX[tid], 20.0]
        zli += [z, z, z]
    g = outputs_grp.create_group(f'Output{out_num}')
    g.create_dataset('mergerTreeIndex', data=np.asarray(tree_order, np.int64))
    g.create_dataset('mergerTreeCount', data=np.full(len(tree_order), NODES_PER_TREE, np.int64))
    nd = g.create_group('nodeData')
    nd.create_dataset('nodeIndex', data=np.asarray(idx, np.int64))
    nd.create_dataset(ParamKeys.is_isolated, data=np.asarray(iso, np.int64))
    nd.create_dataset(ParamKeys.mass_bound, data=np.asarray(bound, np.float64))
    nd.create_dataset(RKEY, data=np.asarray(rmax, np.float64))
    nd.create_dataset(ParamKeys.z_lastisolated, data=np.asarray(zli, np.float64))


def _make_file(path, orders):
    with h5py.File(path, 'w') as f:
        outs = f.create_group('Outputs')
        n = len(orders)
        for k, (out_num, order) in enumerate(sorted(orders.items())):
            _write_output(outs, out_num, order, z=float(n - 1 - k))
    return path


@pytest.fixture
def gfile(tmp_path):
    return _make_file(tmp_path / 'g.hdf5', {1: [1, 2], 2: [1, 2]})


def _only_cache(tmp_path):
    pkls = list(tmp_path.glob('*.pkl'))
    assert len(pkls) == 1, pkls
    return pkls[0]


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _dump(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def test_cache_written_with_version(gfile, tmp_path):
    with h5py.File(gfile, 'r') as f:
        res = subhalo_timeseries(f, 0)
    cached = _load(_only_cache(tmp_path))
    assert isinstance(cached, dict)
    assert cached['subscript_version'] == _subscript_version()
    assert isinstance(cached['result'], dict)
    assert set(cached['result']) == set(res)       # same node ids
    assert SAT_INDEX in res


def test_matching_version_is_reused(gfile, tmp_path):
    with h5py.File(gfile, 'r') as f:
        subhalo_timeseries(f, 0)                    # create cache
        cache = _only_cache(tmp_path)
        obj = _load(cache)
        obj['result'] = {'SENTINEL': 123}          # keep current version
        _dump(cache, obj)
        res = subhalo_timeseries(f, 0)             # should be a cache hit
    assert res == {'SENTINEL': 123}


def test_version_mismatch_regenerates(gfile, tmp_path):
    with h5py.File(gfile, 'r') as f:
        subhalo_timeseries(f, 0)
        cache = _only_cache(tmp_path)
        _dump(cache, {'subscript_version': '0.0.0-stale', 'result': {'SENTINEL': 1}})
        res = subhalo_timeseries(f, 0)             # stale -> recompute + overwrite
    assert 'SENTINEL' not in res
    assert SAT_INDEX in res
    assert _load(cache)['subscript_version'] == _subscript_version()


def test_old_unversioned_format_regenerates(gfile, tmp_path):
    with h5py.File(gfile, 'r') as f:
        subhalo_timeseries(f, 0)
        cache = _only_cache(tmp_path)
        _dump(cache, {999: 'sentinel'})            # legacy raw-dict cache
        res = subhalo_timeseries(f, 0)
    assert res.get(999) != 'sentinel'
    reloaded = _load(cache)
    assert 'subscript_version' in reloaded
    assert reloaded['subscript_version'] == _subscript_version()
    assert SAT_INDEX in reloaded['result']


def test_refresh_recomputes_even_on_match(gfile, tmp_path):
    with h5py.File(gfile, 'r') as f:
        subhalo_timeseries(f, 0)
        cache = _only_cache(tmp_path)
        _dump(cache, {'subscript_version': _subscript_version(), 'result': {'SENTINEL': 1}})
        res = subhalo_timeseries(f, 0, refresh=True)
    assert 'SENTINEL' not in res
    assert SAT_INDEX in res
