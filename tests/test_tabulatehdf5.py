#!/usr/bin/env python
import h5py
import numpy as np
from subscript.tabulatehdf5 import tabulate_trees

def test_tabulate():
    path_dmo = "tests/data/test.hdf5"
    gout = h5py.File(path_dmo)
    trees = tabulate_trees(gout)
    
    total_count = 0
    for tree in trees: 
        total_count += len(tree["basicMass"])

    assert(total_count == np.sum(gout["Outputs"]["Output1"]["mergerTreeCount"][:]))    