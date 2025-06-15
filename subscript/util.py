#!/usr/bin/env python3
import warnings
import numpy as np

def deprecated(reason):
    def decorator(func):
        def wrapper(*args, **kwargs):
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
