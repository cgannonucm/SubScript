#!/usr/bin/env python
import numpy as np
from numpy import testing

from subscript.defaults import ParamKeys
from subscript.scripts.histograms import spatial3d_dn, spatial3d_dndv, spatial2d_dn, spatial2d_dnda, massfunction


def test_3d_dn():
    test_x = np.asarray((0.0, 0.25, 0.5      , 0.7       , 0.8      , 1.3, 1.4))
    test_y = np.asarray((0.0, 0.00, 0.5      , 0.3       , 0.9      , 0.0, 0.0))
    test_z = np.asarray((0.0, 0.00, 0.5      , 0.4       , 0.1      , 0.0, 0.0))

    r      = np.asarray((0.0, 0.25, 0.8660254, 0.86023253, 1.2083046, 1.3, 1.4)) 

    mockdata = {
                    ParamKeys.x: test_x,
                    ParamKeys.y: test_y,
                    ParamKeys.z: test_z
    } 
    
    bins = np.linspace(0,1.5,4)

    dn, dn_r = spatial3d_dn(mockdata, bins=bins)
    
    out_expected = np.asarray((2, 2, 3),dtype=int)

    testing.assert_allclose(dn_r, bins        )
    testing.assert_allclose(dn  , out_expected)

def test_2d_dn():
    test_x = np.asarray((0.0, 0.25, 0.5       , 0.7       , 0.8       , 1.3, 1.4))
    test_y = np.asarray((0.0, 0.00, 0.5       , 0.3       , 0.9       , 0.0, 0.0))
    test_z = np.asarray((0.0, 0.00, 0.5       , 0.4       , 0.1       , 0.0, 0.0))

    r_xy   = np.asarray((0.0, 0.25, 0.70710678, 0.76157731, 1.20415946, 1.3, 1.4 ))
    r_xz   = np.asarray((0.0, 0.25, 0.70710678, 0.80622577, 0.80622577, 1.3, 1.4 ))
    r_yz   = np.asarray([0.0, 0.00, 0.70710678, 0.5       , 0.90553851, 0.0, 0.0])

    mockdata = {
                    ParamKeys.x: test_x,
                    ParamKeys.y: test_y,
                    ParamKeys.z: test_z
    }
    
    bins = np.linspace(0,1.5,4)

    norm_xy = np.asarray((0, 0, 1))
    norm_xz = np.asarray((0, 1, 0))
    norm_yz = np.asarray((1, 0, 0))

    out_expected_xy = np.asarray((2, 2, 3), dtype=int)
    out_expected_xz = np.asarray((2, 3, 2), dtype=int)
    out_expected_yz = np.asarray((4, 3, 0), dtype=int)

    dn_xy, dn_r = spatial2d_dn(mockdata, bins=bins, normvector=norm_xy)
    dn_xz, dn_r = spatial2d_dn(mockdata, bins=bins, normvector=norm_xz)
    dn_yz, dn_r = spatial2d_dn(mockdata, bins=bins, normvector=norm_yz)

    testing.assert_allclose(dn_r, bins        )

    testing.assert_allclose(dn_xy, out_expected_xy)
    testing.assert_allclose(dn_xz, out_expected_xz)
    testing.assert_allclose(dn_yz, out_expected_yz)

def test_3d_dndv():
    n = 10000
    test_x = np.linspace(0.1, 1.5, n - 1)
    test_y = np.zeros(n -1)
    test_z = np.zeros(n - 1)

    mockdata = {
                    ParamKeys.x: test_x,
                    ParamKeys.y: test_y,
                    ParamKeys.z: test_z
    } 
    
    bins = np.linspace(0.1,1.5, n)

    dndv, dn_r = spatial3d_dndv(mockdata, bins=bins)
    testing.assert_allclose(dn_r, bins        )

    hist_expected = np.ones(n - 1)
    dv = 4 * np.pi * ((bins[1:] + bins[:-1]) / 2)**2 * (bins[1:] - bins[:-1])

    out_expected = hist_expected / dv

    testing.assert_allclose(dndv, out_expected, rtol=1E-3)

def test_2d_dnda():
    n = 10000
    test_x = np.zeros(n - 1)
    test_y = np.linspace(0.1, 1.5, n - 1)
    test_z = np.zeros(n - 1)

    mockdata = {
                    ParamKeys.x: test_x,
                    ParamKeys.y: test_y,
                    ParamKeys.z: test_z
    } 
    
    bins = np.linspace(0.1,1.5, n)

    dndv, dn_r = spatial2d_dnda(mockdata,normvector=np.asarray((1,0,0)), bins=bins)
    testing.assert_allclose(dn_r, bins        )

    hist_expected = np.ones(n - 1)
    da = 2 * np.pi * ((bins[1:] + bins[:-1]) / 2) * (bins[1:] - bins[:-1])

    out_expected = hist_expected / da

    testing.assert_allclose(dndv, out_expected, rtol=1E-3)

def test_massfunction():
    mockdata = {
                    ParamKeys.mass_basic: np.array((1,1,1,1,3)) 
    }
    bins = np.linspace(0,4,3)
    mf_actual, mf_actual_bins = massfunction(mockdata, key_mass=ParamKeys.mass_basic, bins=bins)    
    mf_expected = np.array((4 / 2, 1 / 2))
    
    testing.assert_allclose(mf_actual_bins, bins)
    testing.assert_allclose(mf_actual, mf_expected)



