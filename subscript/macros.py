#!/usr/bin/env python
from typing import Iterable, Callable
import numpy as np
import h5py
from copy import copy
from subscript.tabulatehdf5 import NodeProperties, tabulate_trees
from subscript.wrappers import freeze
from datetime import datetime
from numpy.dtypes import StringDType

def macro_add(macros:dict[str, Callable], macro, label=None, **kwargs): 
    if macros.get(label) is not None:
        raise RuntimeError("Macro entry already exists!")
    _m = copy(macros)   
    _m[label] = freeze(macro, **kwargs)
    return _m



def macro_run_file(gout, macros, statfuncs):
    return {key:func(gout, summarize=True, statfuncs=statfuncs)  for key, func in macros.items()}

def macro_runner_def(gouts, macros, statfuncs):
    return [(gout.filename, macro_run_file(gout, macros, statfuncs)) for gout in gouts]

def gen_macro_runner(runner):
    def macro_runner(macros:dict[str, Callable],  gouts:Iterable[(h5py.File)], statfuncs)->dict:
        results = runner(gouts, macros, statfuncs)
        macro_results = {key:val for key, val in results}
        out = {}

        for _id, vals in macro_results.items():
            out[_id] = {}
            for key, val in vals.items():
                sfs = [np.mean, ] if statfuncs is None else statfuncs 
                # For clarity, split into seperate entries for each stat function
                for sf, v in zip(sfs, val):
                    out[_id][f"{key} ({sf.__name__})"] = v
        return out
    return macro_runner

def macro_run(macros:dict[str, tuple[Callable, str]], 
                gouts:Iterable[(h5py.File)], 
                statfuncs=None,runner=None):

    _run = gen_macro_runner(macro_runner_def) if runner is None else runner

    macro_results = _run(macros, gouts, statfuncs) 

    # Create initial dictionary
    entry_fname = next(macro_results.__iter__())
    entry = macro_results[entry_fname]  
    nouts = len(macro_results)
    
    out = {i : {} for i in entry}

    ids = np.asarray(list(macro_results.keys()))
    out["id"] = {"out0": np.asarray([key.encode("ascii", "ignore") for key in macro_results.keys()])}

    for id, val in macro_results.items():
        n = np.where(id.encode("ascii", "ignore") == out["id"]["out0"])[0][0]

        for key, val in val.items():
            # Handles Single output
            _val = val
            if (not isinstance(val, Iterable)) or isinstance(val, np.ndarray):
                _val = [val, ]

            # Handle multiple outputs
            for nval, v in enumerate(_val): 
                key_out = f"out{nval}"
                if out[key].get(key_out) is None:
                    _shape = (nouts, *v.shape) if isinstance(v, np.ndarray) else nouts
                    out[key][key_out] = np.zeros(_shape)

                out[key][key_out][n] = v
    return out
            
            

def macro_write_out_hdf5(f:h5py.File, macro_out, notes=None, stamp_date = True):
    for key, val in macro_out.items():
        if isinstance(val, dict):
            grp = f.create_group(key)
            for _key, _val in val.items():
                grp.create_dataset(_key, data=_val)
            continue
        f.create_dataset(key, data=val)
    
    now = datetime.now()
    f.attrs["date"] = now.strftime("%m/%d/%Y, %H:%M:%S")
    f.attrs["notes"] = str(notes)