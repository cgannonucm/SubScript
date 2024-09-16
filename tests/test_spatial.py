#!/usr/bin/env python

import numpy as np
from numpy import testing

from subscript.scripts.spatial import project2d, project3d
from subscript.defaults import ParamKeys


def test_project3d():
    test_x       = np.asarray((0.0, 0.25, 0.5       , 0.7       , 0.8        , 1.3, 1.4))
    test_y       = np.asarray((0.0, 0.00, 0.5       , 0.3       , 0.9        , 0.0, 0.0))
    test_z       = np.asarray((0.0, 0.00, 0.5       , 0.4       , 0.1        , 0.0, 0.0))

    mockdata = {
                    ParamKeys.x: test_x,
                    ParamKeys.y: test_y,
                    ParamKeys.z: test_z
    }


    r_actual   = project3d(mockdata) 
    r_expected = np.asarray((0.0, 0.25, 0.8660254, 0.86023253, 1.2083046, 1.3, 1.4)) 
    
    testing.assert_allclose(r_actual, r_expected)

def test_project2d():
    test_x       = np.asarray((0.0, 0.25, 0.5       , 0.7       , 0.8        , 1.3, 1.4))
    test_y       = np.asarray((0.0, 0.00, 0.5       , 0.3       , 0.9        , 0.0, 0.0))
    test_z       = np.asarray((0.0, 0.00, 0.5       , 0.4       , 0.1        , 0.0, 0.0))

    mockdata = {
                    ParamKeys.x: test_x,
                    ParamKeys.y: test_y,
                    ParamKeys.z: test_z
    }

    r_xy_expected = np.asarray((0.0, 0.25, 0.70710678, 0.76157731, 1.20415946, 1.3, 1.4))
    r_xz_expected = np.asarray((0.0, 0.25, 0.70710678, 0.80622577, 0.80622577, 1.3, 1.4))
    r_yz_expected = np.asarray((0.0, 0.00, 0.70710678, 0.5       , 0.90553851, 0.0, 0.0))

    r_xy_actual   = project2d(mockdata, normvector=np.asarray((0,0,1)))
    r_xz_actual   = project2d(mockdata, normvector=np.asarray((0,1,0)))
    r_yz_actual   = project2d(mockdata, normvector=np.asarray((1,0,0)))

    testing.assert_allclose(r_xy_actual, r_xy_expected)
    testing.assert_allclose(r_xz_actual, r_xz_expected)
    testing.assert_allclose(r_yz_actual, r_yz_expected)
