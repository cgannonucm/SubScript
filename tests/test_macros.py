import numpy as np
import os
from numpy import testing
import h5py

from subscript.wrappers import freeze
from subscript.scripts.nodes import nodedata
from subscript.defaults import ParamKeys
from subscript.scripts.nfilters import hosthalos
from subscript.macros import macro_run, macro_write_out_hdf5
from subscript.scripts.histograms import massfunction

def test_macro_run():
    path_dmo = "tests/data/test.hdf5"
    path_dmo2 = "tests/data/test-copy.hdf5"
    gout = h5py.File(path_dmo)
    gout2 = h5py.File(path_dmo2)
    
    massfunction_bins = np.logspace(10, 11, 5)
    
    macros = {
                'haloMass'    : freeze(nodedata, key=ParamKeys.mass_basic, nfilter=hosthalos),
                'z'           : freeze(nodedata, key=ParamKeys.z_lastisolated, nfilter=hosthalos),
                'haloMass, z' : freeze(nodedata, key=(ParamKeys.mass_basic, ParamKeys.z_lastisolated), nfilter=hosthalos),
                'massfunction': freeze(massfunction, bins=massfunction_bins)
    }
      
    out_actual = macro_run(macros, [gout, gout2], statfuncs=[np.mean, np.std])  

    out_expected = {
                    'haloMass (mean)'     : {"out0": np.array((1.e+13, 1.e+13))}, 
                    'haloMass (std)'      : {"out0": np.array((0., 0.))}, 
                    'z (mean)'            : {"out0": np.array((0.5, 0.5))}, 
                    'z (std)'             : {"out0": np.array((0., 0.))}, 
                    'haloMass, z (mean)'  : {
                                              "out0": np.array((1.e+13, 1e+13)), 
                                              "out1": np.array((0.5   , 0.5  ))
                                            },
                    'haloMass, z (std)'   : {
                                              "out0": np.array((0.0, 0.0)), 
                                              "out1": np.array((0.0, 0.0))
                                            },
                    'massfunction (mean)' : {
                                             "out0": np.array(((3.471486e-09, 1.250260e-09, 4.513015e-10, 1.517817e-10),
                                                               (3.471486e-09, 1.250260e-09, 4.513015e-10, 1.517817e-10))),
                                             "out1": np.array(((1.000000e+10, 1.778279e+10, 3.162278e+10, 5.623413e+10, 1.000000e+11),                  
                                                               (1.000000e+10, 1.778279e+10, 3.162278e+10, 5.623413e+10, 1.000000e+11)))
                                            },
                    'massfunction (std)' :  {
                                             "out0": np.array([[6.472391e-10, 2.546976e-10, 1.191253e-10, 6.928095e-11],
                                                               [6.472391e-10, 2.546976e-10, 1.191253e-10, 6.928095e-11]]),
                                             "out1": np.array(((0.000000e+00, 1.907349e-05, 2.670288e-05, 6.866455e-05, 0.000000e+00),
                                                               (0.000000e+00, 1.907349e-05, 2.670288e-05, 6.866455e-05, 0.000000e+00)))
                                            }
                   } 
    
    for key, val,  in out_expected.items():
        for _key, _val in val.items():
            testing.assert_allclose(out_actual[key][_key], _val, rtol=1E-6)

def test_macro_out_hdf5():
    path_dmo = "tests/data/test.hdf5"
    path_dmo2 = "tests/data/test-copy.hdf5"
    gout = h5py.File(path_dmo)
    gout2 = h5py.File(path_dmo2)
    
    macros = {
                "haloMass"    : freeze(nodedata, key=ParamKeys.mass_basic, nfilter=hosthalos),
                "z"           : freeze(nodedata, key=ParamKeys.z_lastisolated, nfilter=hosthalos),
                "haloMass, z" : freeze(nodedata, key=(ParamKeys.mass_basic, ParamKeys.z_lastisolated), nfilter=hosthalos),
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


