
import collections
import numpy as np
import sympy as sy

def is_iterable(x):
    return isinstance(x, collections.Iterable)

def is_symbolic(x):
    return (isinstance(x, sy.expr.Expr) or
            is_iterable(x) and any(is_symbolic(xx) for xx in x))
