import numpy as np
import h5py
from numpy import testing

from subscript.nfilters import nfilter_virialized, nfilter_halos, nfilter_subhalos, nfand, nfor, nfnot
from subscript.defaults import  ParamKeys

def test_nfilter_halos():
    mockdata = {
                    "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
                }
    out_actual = nfilter_halos(mockdata)
    out_expected = np.asarray((True, False, True, True, False), dtype=bool)
    
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_subhalos():
    mockdata = {
                    "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
                }
    out_actual = nfilter_subhalos(mockdata)
    out_expected = np.asarray((False, True, False, False, True), dtype=bool)
    
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_logical_and():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }
 
    test = nfand(nfilter_halos, nfilter_subhalos)   
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = nfand(nfilter_halos, nfilter_subhalos(mockdata))   
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = nfand(nfilter_halos(mockdata), nfilter_subhalos)   
    out_actual = test(mockdata)
    out_expected = np.zeros(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_logical_and():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }
 
    test = nfor(nfilter_halos, nfilter_subhalos)   
    out_actual = test(mockdata)
    out_expected = np.ones(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = nfor(nfilter_halos, nfilter_subhalos(mockdata))   
    out_actual = test(mockdata)
    out_expected = np.ones(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

    test = nfor(nfilter_halos(mockdata), nfilter_subhalos)   
    out_actual = test(mockdata)
    out_expected = np.ones(5, dtype=bool)
    testing.assert_equal(out_actual, out_expected)

def test_nfilter_logical_not():
    mockdata = {
                "nodeIsIsolated": np.asarray((1.0, 0.0, 1.0, 1.0, 0.0))
               }
 
    test = nfnot(nfilter_halos)   
    out_actual = test(mockdata)
    out_expected = np.logical_not(np.asarray((1.0, 0.0, 1.0, 1.0, 0.0)))
    testing.assert_equal(out_actual, out_expected)

    test = nfnot(nfilter_halos(mockdata))   
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
    out_rv       = nfilter_virialized(mockdata)
    out_expected = np.array((True, False, False, True, True)) 
    testing.assert_equal(out_rv, out_expected)

