import numpy as np
import os
from numpy import testing
import h5py


from subscript.wrappers import freeze
from subscript.scripts.nodes import nodedata
from subscript.defaults import ParamKeys
from subscript.scripts.nfilters import nfilter_halos
from subscript.macros import macro_run, macro_write_out_hdf5

def test_macro_run():
    path_dmo = "tests/data/test.hdf5"
    path_dmo2 = "tests/data/test-copy.hdf5"
    gout = h5py.File(path_dmo)
    gout2 = h5py.File(path_dmo2)
    
    macros = {
                "haloMass"    : freeze(nodedata, key=ParamKeys.mass_basic, nfilter=nfilter_halos),
                "z"           : freeze(nodedata, key=ParamKeys.z_lastisolated, nfilter=nfilter_halos),
                "haloMass, z" : freeze(nodedata, key=(ParamKeys.mass_basic, ParamKeys.z_lastisolated), nfilter=nfilter_halos),
    }
      
    out_actual = macro_run(macros, [gout, gout2], statfuncs=[np.mean, np.std])  

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
    
    for key, val,  in out_expected.items():
        for _key, _val in val.items():
            testing.assert_allclose(out_actual[key][_key], _val)

def test_macro_out_hdf5():
    path_dmo = "tests/data/test.hdf5"
    path_dmo2 = "tests/data/test-copy.hdf5"
    gout = h5py.File(path_dmo)
    gout2 = h5py.File(path_dmo2)
    
    macros = {
                "haloMass"    : freeze(nodedata, key=ParamKeys.mass_basic, nfilter=nfilter_halos),
                "z"           : freeze(nodedata, key=ParamKeys.z_lastisolated, nfilter=nfilter_halos),
                "haloMass, z" : freeze(nodedata, key=(ParamKeys.mass_basic, ParamKeys.z_lastisolated), nfilter=nfilter_halos),
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

    with h5py.File("tests/out/test_macro_out.hdf5") as f:
        for key, val,  in out_expected.items():
            for _key, _val in val.items():
                testing.assert_allclose(f[key][_key][:], _val)       


