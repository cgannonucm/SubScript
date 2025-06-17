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
    """
    Add a new "script" to an existing macros dictionary.

    Parameters
    ----------
    macros : dict[str, Callable]
        Dictionary of existing macro functions, keyed by their labels.
    macro : Callable
        The macro function to add.
    label : str, optional
        The label under which to add the macro. If None, uses macro.__name__ or other identifier.
    **kwargs
        Additional keyword arguments passed to `freeze` to partially apply or customize the macro.

    Returns
    -------
    dict[str, Callable]
        A new dictionary containing the old macros plus the newly added frozen macro.

    Raises
    ------
    RuntimeError
        If the label already exists in the macros dictionary.
    """
    if macros.get(label) is not None:
        raise RuntimeError("Macro entry already exists!")
    _m = copy(macros)   
    _m[label] = freeze(macro, **kwargs)
    return _m

def macro_run_file(gout, macros, statfuncs):
    """
    Run a set of macro functions on a single Galacticus output.

    Parameters
    ----------
    gout : NodeProperties
        Dictionary-like structure containing Galacticus output node data.
    macros : dict[str, Callable]
        Dictionary of macro functions to run.
    statfuncs : list[Callable]
        Statistical functions to summarize macro results.

    Returns
    -------
    dict[str, any]
        Dictionary with macro labels as keys and results of running the macros as values.
    """
    return {key:func(gout, summarize=True, statfuncs=statfuncs)  for key, func in macros.items()}

def macro_runner_def(gouts, macros, statfuncs):
    """
    Run macros over multiple Galacticus outputs and gather results.

    Parameters
    ----------
    gouts : Iterable[h5py.File]
        Iterable of Galacticus output files.
    macros : dict[str, Callable]
        Dictionary of macro functions to run.
    statfuncs : list[Callable]
        Statistical functions to summarize macro results.

    Returns
    -------
    list[tuple[str, dict]]
        List of tuples where the first element is the filename string and the second
        element is a dictionary of macro results for that file.
    """
    return [(gout.filename, macro_run_file(gout, macros, statfuncs)) for gout in gouts]

def macro_gen_runner(runner):
    """
    Generate a macro runner function that runs macros on Galacticus outputs
    and formats the results with statistical summaries.

    Parameters
    ----------
    runner : Callable
        A runner function that executes macros on multiple Galacticus outputs.

    Returns
    -------
    Callable
        A function that runs macros and returns a nested dictionary of summarized results.
    """
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

    """
    Run macros on multiple Galacticus outputs, summarizing results.

    Parameters
    ----------
    macros : dict[str, tuple[Callable, str]]
        Dictionary of macros with associated metadata (function and string).
    gouts : Iterable[h5py.File]
        Iterable of Galacticus output files.
    statfuncs : list[Callable], optional
        Statistical functions to summarize results. Defaults to [np.mean].
    runner : Callable, optional
        Custom macro runner function. If None, defaults to internal runner.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Dictionary keyed by output name containing arrays of macro results across inputs.
    """
    _run = macro_gen_runner(macro_runner_def) if runner is None else runner

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
    """
    Write macro results into an HDF5 file, including optional notes and timestamp.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file object to write the results into.
    macro_out : dict
        Dictionary of macro output data to write to the file.
    notes : str, optional
        Optional notes or metadata to store in the file attributes.
    stamp_date : bool, optional
        Whether to write the current datetime as a file attribute. Default is True.

    Returns
    -------
    None
    """
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
