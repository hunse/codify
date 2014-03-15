import numpy as np

import sympy

from converter import OCL_Function
from converter import Return, Block


def _gen_vector(base, n):
    return sympy.symbols(''.join(base + '[%d],' % i for i in xrange(n)))


def _test(function, expression, **converter_args):
    converter = OCL_Function(function, **converter_args)
    assert converter.sympy == expression


def test_identity():
    x = _gen_vector('x', 1)
    _test(lambda x: x, Block([Return(x)]))


def test_vector_identity(n=3):
    x = _gen_vector('x', n)
    _test(lambda x: x, Block([Return(x)]), in_dims=n)


def test_exp(n=3):
    x = _gen_vector('x', n)
    y = map(sympy.exp, x)
    _test(lambda x: np.exp(x), Block([Return(y)]), in_dims=n)
