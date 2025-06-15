#!/usr/bin/env python3
import warnings

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
