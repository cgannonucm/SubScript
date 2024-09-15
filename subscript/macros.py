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

def macro_runner_def(macros:dict[str, Callable],  gouts:Iterable[(h5py.File)], statfuncs)->dict:
    _gouts = [tabulate_trees(o) for o in gouts]
    out = {}

    for o, trees in zip(gouts, _gouts):
        out[o.filename] = {}
        for key, func in macros.items():
            sfs = [np.mean, ] if statfuncs is None else statfuncs 
            vals = func(trees, summarize=True, statfuncs=sfs)
            # For clarity, split into seperate entries for each stat function
            for sf, val in zip(sfs, vals):
                out[o.filename][f"{key} ({sf.__name__})"] = val
 
    return out

def macro_run(macros:dict[str, tuple[Callable, str]], 
                gouts:Iterable[(h5py.File)], 
                statfuncs=None, runner=None):

    _run = macro_runner_def if runner is None else runner

    macro_results = _run(macros, gouts, statfuncs)

    # Create initial dictionary
    entry_fname = next(macro_results.__iter__())
    entry = macro_results[entry_fname]  
    nouts = len(macro_results)

    out = {key: np.zeros((nouts, *np.asarray(val).shape)) for key, val in entry.items()} 
    #out["id"] = np.asarray(list(macro_results.keys()), dtype="utf-8") 
    ids = np.asarray(list(macro_results.keys()))
    out["id"] = np.asarray([key.encode("ascii", "ignore") for key in macro_results.keys()])


    for id, val in macro_results.items():
        n = np.where(id.encode("ascii", "ignore") == out["id"])[0][0]

        for key, arr in val.items():
            out[key][n] = arr
    
    return out

def macro_write_out_hdf5(f:h5py.File, macro_out, notes=None, stamp_date = True):
    for key, val in macro_out.items():
        f.create_dataset(key, data=val)
    now = datetime.now()
    f.attrs["date"] = now.strftime("%m/%d/%Y, %H:%M:%S")
    f.attrs["notes"] = str(notes)


if __name__ == "__main__":
    main()