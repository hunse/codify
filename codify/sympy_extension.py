
import sympy as sy


### functions

copysign = sy.Function('copysign')
expm1 = sy.Function('expm1')
hypot = sy.Function('hypot')
isfinite = sy.Function('isfinite')
isinf = sy.Function('isinf')
isnan = sy.Function('isnan')
ldexp = sy.Function('ldexp')
log1p = sy.Function('log1p')
nextafter = sy.Function('nextafter')
signbit = sy.Function('signbit')

class IfExp(sy.Expr):
    precedence = 1

    def __init__(self, cond, true, false):
        self.cond = cond
        self.true = true
        self.false = false

    def __str__(self):
        return "%s if %s else %s" % (self.cond, self.true, self.false)
