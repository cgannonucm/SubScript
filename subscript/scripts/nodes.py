#!/usr/bin/env python
import numpy as np
from typing import Iterable
from subscript.wrappers import gscript
from subscript.tabulatehdf5 import NodeProperties

@gscript
def nodedata(gout:NodeProperties, key:(str | Iterable[str]), **kwargs):
    return gout[key]

@gscript
def nodecount(gout:NodeProperties, **kwargs):
    return np.sum(gout.get_filter())
