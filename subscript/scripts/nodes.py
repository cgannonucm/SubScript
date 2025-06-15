#!/usr/bin/env python
import numpy as np
from typing import Iterable
from subscript.wrappers import gscript
from subscript.tabulatehdf5 import NodeProperties

@gscript
def nodedata(gout:NodeProperties, key:(str | Iterable[str]), **kwargs):
    """
    Extract one or more fields from the Galacticus node data.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.

    key : str or iterable of str
        Key or list of keys to extract from `gout`.

    Returns
    -------
    np.ndarray or dict of np.ndarray
        The data corresponding to the specified key(s).
    """
    return gout[key]

@gscript
def nodecount(gout:NodeProperties, **kwargs):
    """
    Count the number of nodes selected by the current filter.

    Parameters
    ----------
    gout : GalacticusNodes
        Galacticus like nodedata.
    Returns
    -------
    int
        Number of nodes that pass the active nodefilter.
    """
    return np.sum(gout.get_filter())
