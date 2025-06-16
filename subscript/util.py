#!/usr/bin/env python3
import warnings
import numpy as np
from subscript.defaults import Meta

def deprecated(reason):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not Meta.disableDepreciatedWarning:
                warnings.warn(
                    f"{func.__name__}() is deprecated: {reason}",
                    category=DeprecationWarning,
                    stacklevel=2
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_arraylike(obj) -> bool:
    try:
        np.asarray(obj)
        return True
    except Exception:
        return False
