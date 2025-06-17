#!/usr/bin/env python
import h5py
import numpy as np
from numpy import testing
import os 

from subscript.defaults import ParamKeys
from subscript.scripts.histograms import  spatial2d_dn, massfunction
from subscript.scripts.nodes import nodedata, nodecount
from subscript.wrappers import freeze, gscript, gscript_proj, multiproj
from subscript.scripts.nfilters import r2d, hosthalos
from subscript.macros import macro_run, macro_write_out_hdf5




def test_tabulate_multi_files():
    # Test the ability to tabulate multiple files
    path_dmo = "tests/data/test.hdf5"
    path_dmo_2 = "tests/data/test-copy.hdf5"
    gout, gout_2 = h5py.File(path_dmo), h5py.File(path_dmo_2)
    mass = nodedata((gout, gout_2), ParamKeys.mass_basic)
        
    tree_count = np.sum(gout["Outputs"]["Output1"]["mergerTreeCount"].shape)

    assert(len(mass) == 2 * tree_count)
    

def test_multi_proj():
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
    norm = (norm_xy, norm_xz, norm_yz)

    out_expected_xy = np.asarray((2, 2, 3), dtype=int)
    out_expected_xz = np.asarray((2, 3, 2), dtype=int)
    out_expected_yz = np.asarray((4, 3, 0), dtype=int)

    out_expected = np.mean(np.asarray((out_expected_xy, out_expected_xz, out_expected_yz)), axis=0)

    out = spatial2d_dn(mockdata, bins=bins, normvector=norm, summarize=True)

    out_actual, dn_r = out

    testing.assert_allclose(dn_r, bins)
    testing.assert_allclose(out_actual, out_expected) 


def test_gscript_proj_wrap():
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

    rmin, rmax = 0.2, 0.71
    normvectors = np.identity(3)

    n_expected = (
                   np.sum((r_xy_expected >= rmin) & (r_xy_expected <= rmax))
                  +np.sum((r_xz_expected >= rmin) & (r_xz_expected <= rmax))
                  +np.sum((r_yz_expected >= rmin) & (r_yz_expected <= rmax))
                ) / 3

    nfproj   = freeze(r2d, rmin=rmin, rmax=rmax)
 
    n_actual = gscript_proj(freeze(nodecount, nfilter=nfproj))(mockdata, summarize=True, normvector=normvectors)

    testing.assert_allclose(n_actual, n_expected)

    normvectors2 = ((1,0,0), (0,1,0), (0,0,1))

    n_actual = gscript_proj(freeze(nodecount, nfilter=nfproj))(mockdata, summarize=True, normvector=normvectors2)

    testing.assert_allclose(n_actual, n_expected)


def test_gscript_proj_wrap_none():
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

    rmin, rmax = 0.2, 0.71
    normvectors = np.identity(3)

    n_expected = (
                   np.sum((r_xy_expected >= rmin) & (r_xy_expected <= rmax))
                  +np.sum((r_xz_expected >= rmin) & (r_xz_expected <= rmax))
                  +np.sum((r_yz_expected >= rmin) & (r_yz_expected <= rmax))
                ) / 3

    nfproj   = freeze(r2d, rmin=rmin, rmax=rmax)

    frozentest = gscript_proj(freeze(nodecount, nfilter=nfproj))(None, summarize=True, normvector=normvectors)
    n_actual = frozentest(mockdata)

    testing.assert_allclose(n_actual, n_expected)

    normvectors2 = ((1,0,0), (0,1,0), (0,0,1))

    n_actual = gscript_proj(freeze(nodecount, nfilter=nfproj))(mockdata, summarize=True, normvector=normvectors2)

    testing.assert_allclose(n_actual, n_expected)

def test_multiproj():
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

    rmin, rmax = 0.2, 0.71
    normvectors = np.identity(3)

    n_expected = (
                   np.sum((r_xy_expected >= rmin) & (r_xy_expected <= rmax))
                  +np.sum((r_xz_expected >= rmin) & (r_xz_expected <= rmax))
                  +np.sum((r_yz_expected >= rmin) & (r_yz_expected <= rmax))
                ) / 3

    nfproj   = freeze(r2d, rmin=rmin, rmax=rmax)
 
    n_actual = multiproj(nodecount, nfilter=nfproj)(mockdata, summarize=True, normvector=normvectors)

    testing.assert_allclose(n_actual, n_expected)


def test_multiproj_file():
    # Test that combining multiple projections is the same as
    # passing multiple projection vectors
    path_dmo    = "tests/data/test.hdf5"
    gout        = h5py.File(path_dmo)

    nfproj   = freeze(r2d, rmin=1E-1, rmax=2E-2)

    nv = np.identity(3)
    n_actual = multiproj(nodecount, nfilter=nfproj)(gout, summarize=True, normvector=nv)

    ncomb = []
    ncomb += nodecount(gout, nfilter=nfproj, normvector=nv[0])
    ncomb = nodecount(gout, nfilter=nfproj, normvector=nv[0])
    ncomb = nodecount(gout, nfilter=nfproj, normvector=nv[0])

    n_expected = np.mean(ncomb)

    testing.assert_allclose(n_actual, n_expected)


def test_multiproj_file():
    # Test that projections are treated as trees in output shape
    path_dmo    = "tests/data/test.hdf5"
    gout        = h5py.File(path_dmo)

    nfproj   = freeze(r2d, rmin=1E-1, rmax=2E-2)

    nprojv = 6

    nv = np.random.random(18).reshape(nprojv,3)
    nproj = multiproj(nodecount, nfilter=nfproj)(gout, normvector=nv)

    ntrees = len(nodecount(gout))

    testing.assert_equal(len(nproj), ntrees * nprojv)





def test_gscript_unfilter():
    # Ensure the expected behaviour occours when calling a script within a script
    # The inner script should recieve the unfilter version of the original data passed
    
    nfilter = np.asarray((True, True, True, False, False, False))
    
    mock_data = {
                    "test": np.arange(6)
                }

    n = 0
    @gscript
    def testscript(gout, **kwargs):
        nonlocal n
        n += 1
        if n > 20:
            return
        assert(nodecount(gout) == 6)
        assert(nodecount(gout, **kwargs) == 3)

        testscript(gout, **kwargs)
    
    testscript(mock_data, nfilter=nfilter) 

def test_autofreeze():
    path_dmo = "tests/data/test.hdf5"
    path_dmo2 = "tests/data/test-copy.hdf5"
    gout = h5py.File(path_dmo)
    gout2 = h5py.File(path_dmo2)
    
    macros = {
                "haloMass"    : nodedata(None, key=ParamKeys.mass_basic, nfilter=hosthalos),
                "z"           : nodedata(None, key=ParamKeys.z_lastisolated, nfilter=hosthalos),
                "haloMass, z" : nodedata(None, key=(ParamKeys.mass_basic, ParamKeys.z_lastisolated), nfilter=hosthalos),
    }


    out_expected = {
                    'haloMass (mean)'   : {"out0": np.array((1.e+13, 1.e+13))}, 
                    'haloMass (std)'    : {"out0": np.array((0., 0.))}, 
                    'z (mean)'          : {"out0": np.array((0.5, 0.5))}, 
                    'z (std)'           : {"out0": np.array((0., 0.))}, 
                    'haloMass, z (mean)': {
                                            "out0": np.array((1.e+13, 1e+13)), 
                                            "out1": np.array((0.5   , 0.5  ))
                                          },
                    'haloMass, z (std)' : {
                                            "out0": np.array((0.0, 0.0)), 
                                            "out1": np.array((0.0, 0.0))
                                          }
                   } 
      
    out_actual = macro_run(macros, [gout, gout2], statfuncs=[np.mean, np.std])  
 
    if not os.path.exists("tests/out"):
        f = os.mkdir("tests/out")
    
    with h5py.File("tests/out/test_macro_out.hdf5", "w") as f:
        macro_write_out_hdf5(f, out_actual, notes=None)
     
    out_actual = macro_run(macros, [gout, gout2], statfuncs=[np.mean, np.std])  
 
    if not os.path.exists("tests/out"):
        f = os.mkdir("tests/out")
    
    with h5py.File("tests/out/test_macro_out.hdf5", "w") as f:
        macro_write_out_hdf5(f, out_actual, notes=None)

    with h5py.File("tests/out/test_macro_out.hdf5") as f:
        for key, val,  in out_expected.items():
            for _key, _val in val.items():
                testing.assert_allclose(f[key][_key][:], _val)       


if __name__ == "__main__":
    test_multi_proj()
