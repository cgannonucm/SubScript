#!/usr/bin/env python
import numpy as np
from copy import copy

def macro_add(macros:dict, func, label, description): 
    if macros.get(label) is not None:
        raise RuntimeError("Macro entry already exists!")
    _m = copy(macros)
    _m[label] = (func, description)
    return _m

def 