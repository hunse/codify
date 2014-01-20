"""
A Printer for generating readable representation of most sympy classes.
"""

from __future__ import print_function, division

import sympy as sy
import sympy_extension as sy2

from sympy.core import S, Rational, Pow, Basic, Mul
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import Integer
from sympy.printing.printer import Printer
from sympy.printing.precedence import precedence, PRECEDENCE

import sympy.mpmath.libmp as mlib
from sympy.mpmath.libmp import prec_to_dps

from sympy.utilities import default_sort_key

from helper import is_iterable, is_symbolic


class OCLPrinter(Printer):
    _default_settings = {
        "order": None,
        "full_prec": "auto",
    }

    _relationals = dict()

    ZERO = '0.0f'
    ONE = '1.0f'

    OUTPUT = '__OUTPUT__'

    def parenthesize(self, item, level):
        if precedence(item) <= level:
            return "(%s)" % self._print(item)
        else:
            return self._print(item)

    def stringify(self, args, sep, level=0):
        return sep.join([self.parenthesize(item, level) for item in args])

    def emptyPrinter(self, expr):
        if isinstance(expr, str):
            return expr
        else:
            raise NotImplementedError("Cannot print %r", expr)
        # elif isinstance(expr, Basic):
        #     if hasattr(expr, "args"):
        #         return repr(expr)
        #     else:
        #         raise
        # else:
        #     return str(expr)

    def _print_Abs(self, expr):
        return "abs(%s)" % self._print(expr)

    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)

        PREC = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < PREC:
                l.extend([sign, "(%s)" % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_And(self, expr):
        return ' && '.join(sorted(self.parenthesize(a, precedence(expr))
                           for a in expr.args))

    def _print_Or(self, expr):
        return ' || '.join(sorted(self.parenthesize(a, precedence(expr))
                           for a in expr.args))

    def _print_Not(self, expr):
        assert len(expr.args) == 1
        return '!%s' % self.parenthesize(expr.args[0], precedence(expr))

    def _print_Basic(self, expr):
        l = [self._print(o) for o in expr.args]
        return expr.__class__.__name__ + "(%s)" % ", ".join(l)

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            item = "%s: %s" % (self._print(key), self._print(d[key]))
            items.append(item)

        return "{%s}" % ", ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_Dummy(self, expr):
        return '_' + expr.name

    def _print_Exp1(self, expr):
        return 'E'

    def _print_Function(self, expr):
        function_map = {
            sy.abs: 'fabs',
            sy.acos: 'acos',
            sy.acosh: 'acosh',
            sy.asin: 'asin',
            sy.asinh: 'asinh',
            sy.atan: 'atan',
            sy.atan2: 'atan2',
            sy.atanh: 'atanh',
            # sy.cbrt: 'cbrt',
            sy.ceiling: 'ceil',
            sy.cos: 'cos',
            sy.cosh: 'cosh',
            sy.erf: 'erf',
            sy.erfc: 'erfc',
            sy.exp: 'exp',
            sy2.expm1, 'expm1',
            sy.floor, 'floor',
            sy2.hypot, 'hypot',


        # raise NotImplementedError()
        # return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_ImaginaryUnit(self, expr):
        raise NotImplementedError()
        # return 'I'

    def _print_Infinity(self, expr):
        return 'INFINITY'

    def _print_Inverse(self, I):
        return "%s / %s" % (
            self.ONE, self.parenthesize(I.arg, PRECEDENCE["Pow"]))

    def _print_list(self, expr):
        return "[%s]" % self.stringify(expr, ", ")

    def _print_MatrixBase(self, expr):
        return expr._format_str(self)
    _print_SparseMatrix = \
        _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self._print(expr.parent) + '[%s, %s]'%(expr.i, expr.j)

    def _print_MatrixSlice(self, expr):
        def strslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return ':'.join(map(self._print, x))
        return (self._print(expr.parent) + '[' +
                strslice(expr.rowslice) + ', ' +
                strslice(expr.colslice) + ']')

    def _print_DeferredVector(self, expr):
        return expr.name

    def _print_Mul(self, expr):
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = list(map(lambda x: self.parenthesize(x, prec), a))
        b_str = list(map(lambda x: self.parenthesize(x, prec), b))

        if len(b) == 0:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            if len(a) == 1 and not (a[0].is_Atom or a[0].is_Add):
                return sign + "%s/" % a_str[0] + '*'.join(b_str)
            else:
                return sign + '*'.join(a_str) + "/%s" % b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)

    def _print_MatMul(self, expr):
        return '*'.join([self.parenthesize(arg, precedence(expr))
            for arg in expr.args])

    def _print_MatAdd(self, expr):
        return ' + '.join([self.parenthesize(arg, precedence(expr))
            for arg in expr.args])

    def _print_NaN(self, expr):
        return 'NAN'

    def _print_NegativeInfinity(self, expr):
        return '-INFINITY'

    def _print_Normal(self, expr):
        raise NotImplementedError("normal distribution")

    def _print_TensorIndex(self, expr):
        return expr._print()

    def _print_TensorHead(self, expr):
        return expr._print()

    def _print_TensMul(self, expr):
        return expr._print()

    def _print_TensAdd(self, expr):
        return expr._print()

    def _print_Pi(self, expr):
        return 'M_PI'

    # def _print_PolyElement(self, poly):
    #     return poly.str(self, PRECEDENCE, "%s**%d", "*")

    # def _print_FracElement(self, frac):
    #     if frac.denom == 1:
    #         return self._print(frac.numer)
    #     else:
    #         numer = self.parenthesize(frac.numer, PRECEDENCE["Add"])
    #         denom = self.parenthesize(frac.denom, PRECEDENCE["Atom"]-1)
    #         return numer + "/" + denom

    # def _print_Poly(self, expr):
    #     terms, gens = [], [ self._print(s) for s in expr.gens ]

    #     for monom, coeff in expr.terms():
    #         s_monom = []

    #         for i, exp in enumerate(monom):
    #             if exp > 0:
    #                 if exp == 1:
    #                     s_monom.append(gens[i])
    #                 else:
    #                     s_monom.append(gens[i] + "**%d" % exp)

    #         s_monom = "*".join(s_monom)

    #         if coeff.is_Add:
    #             if s_monom:
    #                 s_coeff = "(" + self._print(coeff) + ")"
    #             else:
    #                 s_coeff = self._print(coeff)
    #         else:
    #             if s_monom:
    #                 if coeff is S.One:
    #                     terms.extend(['+', s_monom])
    #                     continue

    #                 if coeff is S.NegativeOne:
    #                     terms.extend(['-', s_monom])
    #                     continue

    #             s_coeff = self._print(coeff)

    #         if not s_monom:
    #             s_term = s_coeff
    #         else:
    #             s_term = s_coeff + "*" + s_monom

    #         if s_term.startswith('-'):
    #             terms.extend(['-', s_term[1:]])
    #         else:
    #             terms.extend(['+', s_term])

    #     if terms[0] in ['-', '+']:
    #         modifier = terms.pop(0)

    #         if modifier == '-':
    #             terms[0] = '-' + terms[0]

    #     format = expr.__class__.__name__ + "(%s, %s"

    #     from sympy.polys.polyerrors import PolynomialError

    #     try:
    #         format += ", modulus=%s" % expr.get_modulus()
    #     except PolynomialError:
    #         format += ", domain='%s'" % expr.get_domain()

    #     format += ")"

    #     return format % (' '.join(terms), ', '.join(gens))

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_Pow(self, expr, rational=False):
        PREC = precedence(expr)

        if expr.exp == S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if expr.exp == -S.Half and not rational:
                return "%s / sqrt(%s)" % (self.ONE, self._print(expr.base))
            if expr.exp == -1:
                return '%s / %s' % (self.ONE, self.parenthesize(expr.base, PREC))

        b = self._print(expr.base)
        e = self._print(expr.exp)
        if expr.exp.is_Integer:
            return 'pown(%s, %s)' % (b, e)
        if expr.exp.is_positive:
            return 'powr(%s, %s)' % (b, e)
        return 'pow(%s, %s)' % (b, e)

    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s**%s' % (self.parenthesize(expr.base, PREC),
                         self.parenthesize(expr.exp, PREC))

    def _print_Integer(self, expr):
        return str(expr.p)

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            return "%s/%s" % (expr.p, expr.q)

    def _print_PythonRational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            return "%d/%d" % (expr.p, expr.q)

    def _print_Fraction(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_mpq(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_Float(self, expr):
        prec = expr._prec
        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1
        rv = mlib.to_str(expr._mpf_, dps, strip_zeros=strip)
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        return rv

    def _print_Relational(self, expr):
        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                           self._relationals.get(expr.rel_op) or expr.rel_op,
                           self.parenthesize(expr.rhs, precedence(expr)))

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if args:
            args = '[%s]' % args
        return '%s(%s)' % (type(s).__name__, args)

    _print_frozenset = _print_set

    # def _print_Sum(self, expr):
    #     def _xab_tostr(xab):
    #         if len(xab) == 1:
    #             return self._print(xab[0])
    #         else:
    #             return self._print((xab[0],) + tuple(xab[1:]))
    #     L = ', '.join([_xab_tostr(l) for l in expr.limits])
    #     return 'Sum(%s, %s)' % (self._print(expr.function), L)

    def _print_Symbol(self, expr):
        return expr.name
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    # def _print_Identity(self, expr):
    #     return "I"

    # def _print_ZeroMatrix(self, expr):
    #     return "0"

    def _print_str(self, expr):
        return expr

    # def _print_tuple(self, expr):
    #     if len(expr) == 1:
    #         return "(%s,)" % self._print(expr[0])
    #     else:
    #         return "(%s)" % self.stringify(expr, ", ")

    # def _print_Tuple(self, expr):
    #     return self._print_tuple(expr)

    # def _print_Transpose(self, T):
    #     return "%s'" % self.parenthesize(T.arg, PRECEDENCE["Pow"])

    # def _print_Uniform(self, expr):
    #     return "Uniform(%s, %s)" % (expr.a, expr.b)

    def _print_Zero(self, expr):
        return self.ZERO

    # def _print_DMP(self, p):
    #     from sympy.core.sympify import SympifyError
    #     try:
    #         if p.ring is not None:
    #             # TODO incorporate order
    #             return self._print(p.ring.to_sympy(p))
    #     except SympifyError:
    #         pass

    #     cls = p.__class__.__name__
    #     rep = self._print(p.rep)
    #     dom = self._print(p.dom)
    #     ring = self._print(p.ring)

    #     return "%s(%s, %s, %s)" % (cls, rep, dom, ring)

    # def _print_DMF(self, expr):
    #     return self._print_DMP(expr)

    # def _print_Object(self, object):
    #     return 'Object("%s")' % object.name

    def _print_IfExp(self, expr):
        return "(%s) ? %s : %s" % (
            self._print(expr.cond), self._print(expr.true),
            self._print(expr.false))

    def _indent(self, string, indent):
        return ''.join([' '] * indent) + string

    def _print_Assign(self, expr, indent=0):
        target = self._print(expr.target)
        value = self._print(expr.value)
        stmt = "%s = %s;" % (target, value)
        return self._indent(stmt, indent)

    def _print_Return(self, expr, indent=0):
        print_return = lambda i, v: self._indent(
            "%s[%d] = %s;" % (self.OUTPUT, i, self._print(v)), indent)
        if is_iterable(expr.value):
            block = [print_return(i, v) for i, v in enumerate(expr.value)]
        else:
            block = [print_return(0, expr.value)]
        return "\n".join(block)

    def _print_Block(self, expr, indent=None):
        if indent is None:
            indent = 0
        else:
            indent = indent + 4

        return "\n".join([self._print(line, indent=indent)
                          for line in expr.lines])

    def _print_IfBlock(self, expr, indent=0):
        strings = ["if (%s) {" % self._print(expr.cond)]

        if expr.true is not None:
            strings.append(self._print(expr.true, indent=indent))

        if expr.false is not None:
            strings.append("} else {")
            strings.append(self._print(expr.false, indent=indent))

        strings.append("}")
        return "\n".join(strings)


def ocl(expr, **settings):
    """Returns the expression as a string of OCL code."""
    p = OCLPrinter(settings)
    s = p.doprint(expr)
    return s
