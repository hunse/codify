
from sympy.core import pi, oo, symbols, Function, Rational
from sympy.functions import exp, log, sqrt, erf, erfc, gamma, loggamma

# from sympy import ccode
from oclcode import oclcode

x, y, z = symbols('x,y,z')
i = symbols('i', integer=True)

def asserteq(a, b):
    if a != b:
        raise AssertionError("%s != %s" % (a, b))


def test_constants():
    assert oclcode(exp(1)) == "M_E"
    assert oclcode(pi) == "M_PI"
    assert oclcode(oo) == "INFINITY"
    assert oclcode(-oo) == "-INFINITY"


def test_functions_basic():
    assert oclcode(abs(i)) == "abs(i)"
    assert oclcode(abs(x)) == "fabs(x)"
    assert oclcode(log(x)) == "log(x)"
    assert oclcode(log(x, 2)) == "log2(x)"
    assert oclcode(log(x, 10)) == "log10(x)"
    assert oclcode(log(x, 3)) == "log(x)/log(3)"


def test_functions_inserted():
    assert oclcode(2**x) == "exp2(x)"
    assert oclcode(10**x) == "exp10(x)"
    assert oclcode(sqrt(x**2 + y**2)) == "hypot(x, y)"


def test_functions_special():
    assert oclcode(erf(x)) == "erf(x)"
    assert oclcode(erfc(x)) == "erfc(x)"
    assert oclcode(gamma(x)) == "tgamma(x)"
    assert oclcode(loggamma(x)) == "lgamma(x)"


def test_pow():
    assert oclcode(x**0.5) == "sqrt(x)"
    assert oclcode(x**Rational(1, 3)) == "cbrt(x)"
    assert oclcode(x**(-1)) == "1.0/x"
    assert oclcode(i**(-1)) == "1.0/i"
    assert oclcode(x**9) == "pown(x, 9)"
    assert oclcode(x**(-3)) == "pown(x, -3)"
    assert oclcode(x**3.2) == "powr(x, 3.2)"
    assert oclcode(x**(-2.7)) == "pow(x, -2.7)"



# def test():
#     # print( oclcode(sqrt(GoldenRatio + 5)) )
#     # print( oclcode([1,2,3]) )
#     print( oclcode(gamma(x)) )



if __name__ == '__main__':
    test_constants()
    test_functions_basic()
    test_functions_inserted()
    test_functions_special()

    # test()
