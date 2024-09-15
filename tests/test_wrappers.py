#!/usr/bin/env python
from subscript.wrappers import freeze


def test_freeze():
    foo = lambda g, a, **k: a
    bar = freeze(foo, a=1)
    assert(bar(None, a=2) == 1) 
    