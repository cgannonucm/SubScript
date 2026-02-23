import numpy as np
import h5py
from numpy import testing

from subscript.scripts.nfilters import withinrv, hosthalos, subhalos, logical_and, logical_or, logical_not, r2d
from subscript.defaults import  ParamKeys

def test_nfilter_halos():
    mockdata = {
                    "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
                }
    out_actual = hosthalos(mockdata)
    out_expected = np.asarray((True, False, True, True, False), dtype=bool)
    
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_subhalos():
    mockdata = {
                    "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
                }
    out_actual = subhalos(mockdata)
    out_expected = np.asarray((False, True, False, False, True), dtype=bool)
    
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_logical_and():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }
 
    test = logical_and(hosthalos, subhalos)
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = logical_and(hosthalos, subhalos(mockdata))
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = logical_and(hosthalos(mockdata), subhalos)
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_logical_or():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }

    test = logical_or(hosthalos, subhalos)
    out_actual = test(mockdata)
    out_expected = np.ones(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = logical_or(hosthalos, subhalos(mockdata))
    out_actual = test(mockdata)
    out_expected = np.ones(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = logical_or(hosthalos(mockdata), subhalos)
    out_actual = test(mockdata)
    out_expected = np.ones(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_logical_not():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }
 
    test = logical_not(hosthalos)
    out_actual = test(mockdata)
    out_expected = np.logical_not(np.asarray((1.0, 0.0, 1.0, 1.0, 0.0)))
    testing.assert_equal(out_actual, out_expected)

    test = logical_not(hosthalos(mockdata))
    out_actual = test(mockdata)
    out_expected = np.logical_not(np.asarray((1.0, 0.0, 1.0, 1.0, 0.0)))
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_virialized():
    # Test selecting halos within virial radius
    mockdata = {
                ParamKeys.mass_basic : np.array((5.0, 1.0, 1.0, 1.0, 1.0)),
                ParamKeys.rvir       : np.array((0.5, 0  , 0  , 0  , 0  )),
                ParamKeys.x          : np.array((0  , 1.0, 0.5, 0.2, 0.1)),
                ParamKeys.y          : np.array((0  , 1.0, 0.5, 0  , 0.1)),
                ParamKeys.z          : np.array((0  , 0  , 0.5, 0  , 0.1))
    }
    
    # Create test
    out_rv       = withinrv(mockdata)
    out_expected = np.array((True, False, False, True, True)) 
    testing.assert_equal(out_rv, out_expected)


def test_nfilter_project2d():
    test_x       = np.asarray((0.0, 0.25, 0.5       , 0.7       , 0.8        , 1.3, 1.4))
    test_y       = np.asarray((0.0, 0.00, 0.5       , 0.3       , 0.9        , 0.0, 0.0))
    test_z       = np.asarray((0.0, 0.00, 0.5       , 0.4       , 0.1        , 0.0, 0.0))

    mockdata = {
                    ParamKeys.x: test_x,
                    ParamKeys.y: test_y,
                    ParamKeys.z: test_z
    }

    r_xy_expected   = np.asarray((0.0, 0.25, 0.70710678, 0.76157731, 1.20415946, 1.3, 1.4))


    rmin, rmax = 0.2, 0.5
    filter_expected = (r_xy_expected > rmin) & (r_xy_expected < rmax)

    filter_actual   = r2d(mockdata, rmin=rmin, rmax=rmax, normvector=np.array((0, 0, 1)))

    testing.assert_equal(filter_actual, filter_expected)


def test_logical_and_varargs():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }

    # Three callable args
    test = logical_and(hosthalos, hosthalos, hosthalos)
    out_actual = test(mockdata)
    out_expected = hosthalos(mockdata)
    testing.assert_equal(out_actual, out_expected)

    # Mix of callables and arrays
    test = logical_and(hosthalos, hosthalos(mockdata), subhalos)
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    # Four args: all True AND hosthalos AND subhalos → all False
    all_true = np.ones(5, dtype=bool)
    test = logical_and(hosthalos, subhalos, all_true)
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)


def test_logical_or_varargs():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }

    # Three callable args: hosthalos OR subhalos OR hosthalos → all True
    test = logical_or(hosthalos, subhalos, hosthalos)
    out_actual = test(mockdata)
    out_expected = np.ones(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    # All-false array OR hosthalos → hosthalos
    all_false = np.zeros(5, dtype=bool)
    test = logical_or(all_false, all_false, hosthalos)
    out_actual = test(mockdata)
    out_expected = hosthalos(mockdata)
    testing.assert_equal(out_actual, out_expected)


def test_logical_and_shape_mismatch():
    a = np.ones(5, dtype=bool)
    b = np.ones(3, dtype=bool)

    test = logical_and(a, a, b)
    try:
        test()
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_logical_or_shape_mismatch():
    a = np.ones(5, dtype=bool)
    b = np.ones(3, dtype=bool)

    test = logical_or(a, a, b)
    try:
        test()
        assert False, "Expected ValueError"
    except ValueError:
        pass
