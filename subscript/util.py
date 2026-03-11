#!/usr/bin/env python3
"""
General-purpose utility helpers used internally by SubScript.

This module provides small, dependency-light helpers that are shared across
several SubScript sub-modules:

- :func:`deprecated` – decorator for marking functions as deprecated.
- :func:`is_arraylike` – predicate that checks whether an object can be
  converted to a :class:`numpy.ndarray`.
"""
import warnings
import numpy as np
from subscript.defaults import Meta

def deprecated(reason):
    """
    Decorator factory that marks a function as deprecated.

    When the decorated function is called, a :class:`DeprecationWarning` is
    issued (unless :attr:`~subscript.defaults.Meta.disableDepreciatedWarning`
    is ``True``).

    Parameters
    ----------
    reason : str
        Human-readable explanation shown in the warning message, typically
        pointing users to the replacement function.

    Returns
    -------
    Callable
        A decorator that wraps the target function and emits a
        :class:`DeprecationWarning` on every call.

    Examples
    --------
    >>> from subscript.util import deprecated
    >>> @deprecated("Use new_func() instead")
    ... def old_func():
    ...     pass
    >>> old_func()  # emits DeprecationWarning: old_func() is deprecated: Use new_func() instead
    """
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
    """
    Return ``True`` if *obj* can be converted to a :class:`numpy.ndarray`.

    This is a lightweight duck-type check: it attempts ``numpy.asarray(obj)``
    and returns ``False`` if any exception is raised.

    Parameters
    ----------
    obj : object
        The object to test.

    Returns
    -------
    bool
        ``True`` if ``numpy.asarray(obj)`` succeeds without raising an
        exception; ``False`` otherwise.
    """
    try:
        np.asarray(obj)
        return True
    except Exception:
        return False
