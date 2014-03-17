"""
OCL code printer

The OCLCodePrinter converts single SymPy expressions into
single OpenCL expressions.
"""

from sympy.core import Add, Function, Pow
from sympy.printing.ccode import CCodePrinter

is_square = lambda x: (isinstance(x, Pow) and x.exp == 2)
is_inv = lambda x: (isinstance(x, Pow) and x.exp == -1)
is_log = lambda x: (isinstance(x, Function) and x.func.__name__ == 'log'
                    and len(x.args) == 1)

known_functions = {
    "Abs": [(lambda x: x.is_integer, "abs"),
            (lambda x: not x.is_integer, "fabs")],
    "gamma": [(lambda x: True, "tgamma")],
    "loggamma": [(lambda x: True, "lgamma")],
}


class OCLCodePrinter(CCodePrinter):

    def __init__(self, settings={}):
        settings = dict(settings)
        userfuncs = settings.setdefault('user_functions', {})
        for k, v in known_functions.items():
            userfuncs.setdefault(k, v)
        CCodePrinter.__init__(self, settings)

    def _print_Infinity(self, expr):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr):
        return '-INFINITY'

    def _print_NaN(self, expr):
        return 'NAN'

    def _print_Mul(self, expr):
        if len(expr.args) == 2: # log2 and log10
            a, b = expr.args
            if is_inv(a) and is_log(a.base) and is_log(b):
                log_base = a.base.args[0]
                if log_base in [2, 10]:
                    return "log%d(%s)" % (log_base, self._print(b.args[0]))
        return CCodePrinter._print_Mul(self, expr)

    def _print_Pow(self, expr):
        if expr.base == 2:
            return "exp2(%s)" % self._print(expr.exp)
        if expr.base == 10:
            return "exp10(%s)" % self._print(expr.exp)
        if expr.exp == 0.5 and isinstance(expr.base, Add):
            args = expr.base.args
            if len(args) == 2 and all(map(is_square, args)):
                if self.order != 'none':
                    args = self._as_ordered_terms(expr.base, order=None)
                    # TODO: (above) is it fine to use `order=None`?
                return "hypot(%s, %s)" % (self._print(args[0].args[0]),
                                          self._print(args[1].args[0]))
        if expr.exp == Rational(1, 3):
            return "cbrt(%s)" % self._print(expr.base)
        if expr.exp.is_integer and expr.exp != -1:
            return "pown(%s, %s)" % (
                self._print(expr.base), self._print(expr.exp))
        if expr.exp.is_positive and expr.exp != 0.5:
            return "powr(%s, %s)" % (
                self._print(expr.base), self._print(expr.exp))
        return CCodePrinter._print_Pow(self, expr)

    # def _print_Function(self, expr):
        # return CCodePrinter._print_Function(self, expr)


def oclcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of OCL code.

    See the documentation for `ccode` for more information.
    """

    return OCLCodePrinter(settings).doprint(expr, assign_to)


def print_oclcode(expr, **settings):
    """Prints OCL representation of the given expression."""
    print(oclcode(expr, **settings))
