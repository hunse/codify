
import __builtin__
import ast
import collections
import inspect
import math

import numpy as np
# import scipy as sp
# import scipy.special as sps
import sympy as sy
import sympy_extension as sy2

from helper import is_iterable, is_symbolic

import ipdb

def strip_leading_whitespace(source):
    lines = source.splitlines()
    assert len(lines) > 0
    first_line = lines[0]
    n_removed = len(first_line) - len(first_line.lstrip())
    if n_removed > 0:
        return '\n'.join(line[n_removed:] for line in lines)
    else:
        return source


class Assign(object):
    def __init__(self, target, value):
        self.target = target
        self.value = value

    def __eq__(self, b):
        return isinstance(b, Assign) and self.target == b.target and self.value == b.value

    def __str__(self):
        return "%s = %s" % (self.target, self.value)

class Return(object):
    def __init__(self, value):
        self.value = value

    def __eq__(self, b):
        return isinstance(b, Return) and all(self.value == b.value)

    def __str__(self):
        return "return %s" % self.value

class Block(object):
    def __init__(self, lines):
        self.lines = lines

    def __eq__(self, b):
        return (isinstance(b, Block) and len(self.lines) == len(b.lines) and
                all([la == lb for la, lb in zip(self.lines, b.lines)]))

    def __str__(self):
        return "\n".join([str(line) for line in self.lines])

class IfBlock(object):
    def __init__(self, cond, true, false):
        self.cond = cond
        self.true = true
        self.false = false

    def __str__(self):
        return "\n".join(["if %s:" % self.cond, str(self.true),
                          "else:", str(self.false)])


class Function_Finder(ast.NodeVisitor):
    # Finds a FunctionDef or Lambda in an Abstract Syntax Tree

    def __init__(self):
        self.fn_node = None

    def generic_visit(self, stmt):
        if isinstance(stmt, ast.Lambda) or isinstance(stmt, ast.FunctionDef):
            if self.fn_node is None:
                self.fn_node = stmt
            else:
                raise NotImplementedError(
                    "The source code associated with the function "
                    "contains more than one function definition")

        super(self.__class__, self).generic_visit(stmt)


class SympyTranslator(ast.NodeVisitor):

    function_list = [ # functions that work on numpy or sympy objects
        enumerate,
        abs, np.abs, np.absolute, np.add, np.asarray,
        np.divide, np.mod, np.multiply, np.negative,
        np.power, np.prod, np.product,
        np.reciprocal, np.remainder,
        np.square, np.subtract,
        ]

    function_map = { # functions that must be mapped from numpy to sympy
        all: lambda x: sy.And(*x),
        any: lambda x: sy.Or(*x),
        max: lambda x: sy.Max(*x),
        min: lambda x: sy.Min(*x),
        math.acos: sy.acos,
        math.acosh: sy.acosh,
        math.asin: sy.asin,
        math.asinh: sy.asinh,
        math.atan: sy.atan,
        math.atan2: sy.atan2,
        math.atanh: sy.atanh,
        math.ceil: sy.ceiling,
        math.copysign: lambda x, y: sy.abs(x) * sy.sign(y),
        math.cos: sy.cos,
        math.cosh: sy.cosh,
        math.degrees: sy.deg,
        math.erf: sy.erf,
        math.erfc: sy.erfc,
        math.exp: sy.exp,
        math.expm1: lambda x: sy.exp(x) - 1.,
        math.fabs: abs,
        math.floor: sy.floor,
        math.fmod: sy.mod,
        math.gamma: sy.gamma,
        math.hypot: lambda x, y: sy.sqrt(x**2 + y**2),
        # math.isinf: 'isinf',
        # math.isnan: 'isnan',
        math.ldexp: lambda x, y: x * 2**y,
        math.lgamma: lambda x: sy.log(abs(sy.gamma(x))),
        math.log: sy.log,
        math.log10: lambda x: sy.log(x, 10),
        math.log1p: lambda x: sy.log(1 + x),
        # math.modf: # TODO: return integer and fractional parts of x
        math.pow: lambda x, y: x**y,
        math.radians: lambda x: sy.rad,
        math.sin: sy.sin,
        math.sinh: sy.sinh,
        math.sqrt: sy.sqrt,
        math.tan: sy.tan,
        math.tanh: sy.tanh,
        np.arccos: sy.acos,
        np.arccosh: sy.acosh,
        np.arcsin: sy.asin,
        np.arcsinh: sy.asinh,
        np.arctan: sy.atan,
        np.arctanh: sy.atanh,
        np.arctan2: sy.atan2,
        # np.bitwise_and: lambda x, y: BinExp(x, '&', y),
        # np.bitwise_not: lambda x: UnaryExp('~', x),
        # np.bitwise_or: lambda x, y: BinExp(x, '|', y),
        # np.bitwise_xor: lambda x, y: BinExp(x, '^', y),
        np.ceil: sy.ceiling,
        # np.copysign: lambda x, y: np.abs(x) * np.sign(y),
        np.copysign: sy2.copysign,
        np.cos: sy.cos,
        np.cosh: sy.cosh,
        np.deg2rad: sy.rad,
        np.degrees: sy.deg,
        np.equal: lambda x, y: Equality(x, y),
        np.exp: sy.exp,
        np.exp2: lambda x: 2**x,
        # np.expm1: lambda x: sy.exp(x) - 1.,
        np.expm1: sy2.expm1,
        np.fabs: abs,
        np.floor: sy.floor,
        np.floor_divide: lambda x, y: sy.floor(x / y),
        np.fmax: sy.Max,
        np.fmin: sy.Min,
        np.fmod: sy.mod,
        np.greater: lambda x, y: x > y,
        np.greater_equal: lambda x, y: x >= y,
        # np.hypot: lambda x, y: sy.sqrt(x**2 + y**2),
        np.hypot: sy2.hypot,
        np.isfinite: sy2.isfinite,
        np.isinf: sy2.isinf,
        np.isnan: sy2.isnan,
        np.ldexp: sy2.ldexp,
        # np.ldexp: lambda x, y: x * 2**y,
        # np.left_shift: lambda x, y: BinExp(x, '<<', y),
        np.less: lambda x, y: x < y,
        np.less_equal: lambda x, y: x <= y,
        np.log: sy.log,
        np.log10: lambda x: sy.log(x, 10),
        # np.log1p: lambda x: sy.log(1 + x),
        np.log1p: sy2.log1p,
        np.log2: lambda x: sy.log(x, 2),
        np.logaddexp: lambda x, y: sy.log(sy.exp(x) + sy.exp(y)),
        np.logaddexp2: lambda x, y: sy.log(2**x + 2**y, 2),
        np.logical_and: sy.And,
        np.logical_not: sy.Not,
        np.logical_or: sy.Or,
        np.logical_xor: sy.Xor,
        np.maximum: sy.Max,
        np.minimum: sy.Min,
        np.nextafter: sy2.nextafter,
        np.rad2deg: sy.deg,
        np.radians: sy.rad,
        np.sign: sy.sign,
        # np.signbit: lambda x: BinExp(x, '<', NumExp(0)),
        np.signbit: sy2.signbit,
        np.sin: sy.sin,
        np.sinh: sy.sinh,
        np.sqrt: sy.sqrt,
        np.tan: sy.tan,
        np.tanh: sy.tanh,
        # sps.gamma: sy.gamma,
        }

    op_map = {
        ast.Add: lambda x, y: x + y,
        ast.And: sy.And,
        ast.Div: lambda x, y: x / y,
        ast.Eq: sy.Eq,
        ast.FloorDiv: lambda x, y: sy.floor(x / y),
        ast.Gt: sy.Gt,
        ast.GtE: sy.Ge,
        ast.Lt: sy.Lt,
        ast.LtE: sy.Le,
        ast.Mod: lambda x, y: x % y,
        ast.Mult: lambda x, y: x * y,
        ast.Not: sy.Not,
        ast.NotEq: sy.Ne,
        ast.Or: sy.Or,
        ast.Pow: lambda x, y: x**y,
        ast.Sub: lambda x, y: x - y,
        }

    MAX_VECTOR_LENGTH = 25
    builtins = __builtin__.__dict__

    def _check_vector_length(self, length):
        if length > self.MAX_VECTOR_LENGTH:
            raise ValueError("Vectors of length >%s are not supported"
                             % self.MAX_VECTOR_LENGTH)

    def _new_symbolic_vector(self, name, length):
        if length > self.MAX_VECTOR_LENGTH:
            raise ValueError("Vectors of length >%s are not supported"
                             % self.MAX_VECTOR_LENGTH)
        else:
            return np.array(sy.symbols(
                    ['%s[%d]' % (name, i) for i in xrange(length)]))

    def __init__(self, source, globals_dict, closure_dict,
                 in_dims=None, out_dim=None):
        self.source = source
        self.globals = globals_dict
        self.closures = closure_dict
        # self.in_dims = in_dims
        self.out_dim = out_dim

        self.locals = {}
        self.temps = {}  # for comprehensions

        ### parse and make code
        a = ast.parse(source)
        ff = Function_Finder()
        ff.visit(a)
        function_def = ff.fn_node

        self.arg_names = [arg.id for arg in function_def.args.args]

        if in_dims is None:
            in_dims = [1] * len(self.arg_names) # assume all scalars
        elif not is_iterable(in_dims):
            in_dims = (in_dims,)
        self.arg_dims = dict(zip(self.arg_names, in_dims))
        for v in self.arg_dims.values():
            self._check_vector_length(v)

        self.args = dict(zip(
                self.arg_names,
                [self._new_symbolic_vector(name, dims)
                 for name, dims in zip(self.arg_names, in_dims)]
                ))

        if isinstance(function_def, ast.FunctionDef):
            self.function_name = function_def.name
            self.body = self.visit_block(function_def.body)
        elif isinstance(function_def, ast.Lambda):
            if hasattr(function_def, 'targets'):
                self.function_name = function_def.targets[0].id
            else:
                self.function_name = "<lambda>"

            r = ast.Return() #wrap lambda expression to look like a one-line function
            r.value = function_def.body
            r.lineno = 1
            r.col_offset = 4
            self.body = self.visit_block([r])
        else:
            raise RuntimeError(
                "Expected function definition or lambda function assignment, "
                "got " + str(type(function_def)))

    # def visit(self, node):
    #     return ast.NodeVisitor.visit(self, node)

    def visit_Name(self, expr):
        name = expr.id
        for d in [self.temps, self.locals, self.args,
                  self.closures, self.globals, self.builtins]:
            if name in d:
                return d[name]

        raise ValueError("Unrecognized name '%s'" % name)

    def visit_Attribute(self, expr):
        if isinstance(expr, ast.Name):
            return self.visit(expr)
        else:
            return getattr(self.visit_Attribute(expr.value), expr.attr)

    def visit_Num(self, expr):
        return expr.n

    def visit_Str(self, expr):
        return expr.s

    def visit_Index(self, expr):
        value = self.visit(expr.value)
        if not isinstance(value, int):
            raise ValueError("Index must be an integer")
        return value

    def visit_Ellipsis(self, expr):
        raise NotImplementedError("Ellipsis")

    def visit_Slice(self, expr):
        def visit(ast):
            if ast is None:
                return None
            else:
                value = self.visit(ast)
                if not isinstance(value, int):
                    raise ValueError("Index must be an integer")
                return value

        lower = visit(expr.lower)
        upper = visit(expr.upper)
        step = visit(expr.step)
        return slice(lower, upper, step)

    def visit_ExtSlice(self, expr):
        raise NotImplementedError("ExtSlice")

    def visit_Subscript(self, expr):
        assert isinstance(expr.value, ast.Name)
        var = self.visit(expr.value)
        s = self.visit(expr.slice)
        return var[s]

    def _name(self, name):
        return ast.Name(name, ast.Load())

    def _eval_ast(self, ast_object, environment, tag='string'):
        def set_line_col(ast_object):
            if isinstance(ast_object, ast.expr):
                ast_object.lineno = 0
                ast_object.col_offset = 0
                for field in ast_object._fields:
                    set_line_col(getattr(ast_object, field))
            elif isinstance(ast_object, list):
                for a in ast_object:
                    set_line_col(a)

        set_line_col(ast_object)
        locals().update(environment)
        return eval(compile(ast.Expression(ast_object),
                            filename='<%s>' % tag, mode='eval'))

    def visit_UnaryOp(self, expr):
        operand = self.visit(expr.operand)
        if is_symbolic(operand):
            op = self.op_map.get(type(expr.op))
            if op is None:
                raise NotImplementedError(
                    "'%r' operator not supported for symbols" % expr.op)
            return op(operand)
        else:
            op = ast.UnaryOp(expr.op, self._name('operand'))
            return self._eval_ast(op, locals(), tag='visit_UnaryOp')

    def visit_BinOp(self, expr):
        left = self.visit(expr.left)
        right = self.visit(expr.right)
        if is_symbolic(left) or is_symbolic(right):
            op = self.op_map.get(type(expr.op))
            if op is None:
                raise NotImplementedError(
                    "'%r' operator not supported for symbols" % expr.op)
            return op(left, right)
        else:
            op = ast.BinOp(self._name('left'), expr.op, self._name('right'))
            return self._eval_ast(op, locals(), tag='visit_BinOp')

    def visit_BoolOp(self, expr):
        op = self.op_map.get(type(expr.op))
        values = [self.visit(v) for v in expr.values]
        return op(*values)

    def visit_Compare(self, expr):
        assert len(expr.ops) == 1
        assert len(expr.comparators) == 1
        op = self.op_map.get(type(expr.ops[0]))
        return op(self.visit(expr.left), self.visit(expr.comparators[0]))

    def visit_Call(self, expr):
        assert expr.kwargs is None, "kwargs not implemented"
        func = self.visit(expr.func)
        args = [self.visit(arg) for arg in expr.args]

        if any(map(is_symbolic, args)):
            if func in self.function_list:
                return func(*args)
            elif func in self.function_map:
                vector_func = np.vectorize(self.function_map[func])
                return vector_func(*args)
            else:
                raise ValueError(
                    "'%s' has no symbolic equivalent" % func.__name__)
        else:
            return func(*args)

    def visit_List(self, expr):
        return [self.visit(elt) for elt in expr.elts]

    def visit_Tuple(self, expr):
        return tuple([self.visit(elt) for elt in expr.elts])

    def visit_Expr(self, expr):
        raise NotImplementedError("Expr")

    def visit_GeneratorExp(self, expr):
        raise NotImplementedError("GeneratorExp")

    def visit_ListComp(self, expr):
        assert len(expr.generators) == 1, "Multiple generators not implemented"
        gen = expr.generators[0]
        iterator = self.visit(gen.iter)
        # if is_symbolic(iterator):
        #     raise NotImplementedError("Symbolic generators not implemented")
        if not is_iterable(iterator):
            raise ValueError("%r is not iterable" % iterator)

        target = gen.target
        if isinstance(target, ast.Name):
            temps = [target.id]
        elif (isinstance(target, ast.Tuple) and
              all(isinstance(e, ast.Name) for e in target.elts)):
            temps = [e.id for e in target.elts]
        else:
            raise ValueError("Unrecognized target for list comprehension")

        for temp in temps:
            assert temp not in self.temps

        result = []
        for i, x in enumerate(iterator):
            if len(temps) == 1:
                self.temps[temps[0]] = x
            else:
                for temp, xj in zip(temps, x):
                    self.temps[temp] = xj
            result.append(self.visit(expr.elt))
            self._check_vector_length(i)

        for temp in temps:
            del self.temps[temp]
        return result

    def visit_IfExp(self, expr):
        cond = self.visit(expr.test)
        true = self.visit(expr.body)
        false = self.visit(expr.orelse)
        if is_iterable(true) or is_iterable(false):
            make = np.vectorize(lambda c, t, f: sy2.IfExp(c, t, f))
            return make(cond, true, false)
        else:
            return sy2.IfExp(cond, true, false)

    def visit_Print(self, expr):
        assert expr.dest is None, "other dests not implemented"
        if (len(expr.values) == 1
            and isinstance(expr.values[0], ast.BinOp)
            and isinstance(expr.values[0].op, ast.Mod)
            and isinstance(expr.values[0].left, ast.Str)):
            # we're using string formatting
            stmt = self.visit(expr.values[0].left)[:-1] + '\\n"'
            if isinstance(expr.values[0].right, ast.Tuple):
                args = [str(self.visit(arg)) for arg in expr.values[0].right.elts]
            else:
                args = [str(self.visit(expr.values[0].right))]
            return ["printf(%s);" % ', '.join([stmt] + args)]
        else:
            stmt = '"' + ' '.join(['%s' for arg in expr.values]) + '\\n"'
            args = ', '.join([str(self.visit(arg)) for arg in expr.values])
            return ["printf(%s, %s);" % (stmt, args)]

    def _visit_lhs(self, lhs):
        if isinstance(lhs, ast.Name):
            name = lhs.id
            unassignables = [self.args, self.closures, self.globals]
            if any(name in d for d in unassignables):
                raise ValueError("Can only assign to a local variable")
            else:
                # if name not in self.init:
                #     # TODO: make new variables of types other than float?
                #     self.init[name] = "float %s;" % name  # make a new variable
                if name not in self.locals:
                    self.locals.append(name)
                return name
        else:
            raise NotImplementedError("Complex LHS")

    def visit_Assign(self, expr):
        assert len(expr.targets) == 1, "Multiple targets not implemented"
        rhs = self.visit(expr.value)
        lhs = self._visit_lhs(expr.targets[0])

        assert isinstance(rhs, Expression), (
            "Can only assign math expressions, not '%s'" % type(rhs))
        return ["%s = %s;" % (lhs, rhs.to_ocl())]

    def visit_AugAssign(self, expr):
        lhs = self._visit_lhs(expr.target)
        rhs = self._visit_binary_op(expr.op, expr.target, expr.value)
        assert isinstance(rhs, Expression), (
            "Can only assign math expressions, not '%s'" % type(rhs))
        return ["%s = %s;" % (lhs, rhs.to_ocl())]

    def visit_Return(self, expr):
        return Return(self.visit(expr.value))

    def visit_If(self, expr):
        test = self.visit(expr.test)
        ifblock = self.visit_block(expr.body)
        elseblock = self.visit_block(expr.orelse)
        return IfBlock(test, ifblock, elseblock)

    def visit_While(self, expr):
        raise NotImplementedError("While")

    def visit_For(self, expr):
        raise NotImplementedError("For")

    def visit_FunctionDef(self, expr):
        raise NotImplementedError("FunctionDef")

    def visit_Lambda(self, expr):
        raise NotImplementedError("Lambda")

    def visit_block(self, stmts):
        if len(stmts) > 0:
            return Block([self.visit(stmt) for stmt in stmts])
        else:
            return None


class OCL_Function(object):
    def __init__(self, fn, in_dims=None, out_dim=None):
        self.fn = fn
        self.in_dims = in_dims
        self.out_dim = out_dim
        self._translator = None

    @staticmethod
    def _is_lambda(v):
        return isinstance(v, type(lambda: None)) and v.__name__ == '<lambda>'

    def _get_ocl_translator(self):
        # if self.fn in direct_funcs or self.fn in indirect_funcs:
        #     assert self.in_dims is not None, (
        #         "Must supply input dimensionality for raw function")
        #     function = self.fn
        #     def dummy(x):
        #         return function(x)
        #     fn = dummy
        # else:
        #     fn = self.fn

        fn = self.fn

        source = inspect.getsource(fn)
        source = strip_leading_whitespace(source)

        globals_dict = fn.func_globals
        closure_dict = (
            dict(zip(fn.func_code.co_freevars,
                     [c.cell_contents for c in fn.func_closure]))
            if fn.func_closure is not None else {})

        return SympyTranslator(source, globals_dict, closure_dict,
                               in_dims=self.in_dims, out_dim=self.out_dim)

    @property
    def translator(self):
        if self._translator is None:
            self._translator = self._get_ocl_translator()
        return self._translator

    @property
    def sympy(self):
        return self.translator.body


if __name__ == '__main__':
    from ocl_printer import ocl

    # def function(x):
    #     return x[:] + 3

    # def function(x):
    #     return x + range(3)

    # def function(x):
    #     if not all([xx > 3 for xx in x]):
    #         return x + 2
    #     else:
    #         return x - 1

    # def function(x):
    #     return [xi + i + 3 for i, xi in enumerate(x[::-1])]

    def function(x):
        # return x if all([xx > 3 for xx in x]) else -1
        return np.exp(x)

    expr = OCL_Function(function, in_dims=3).sympy
    # print expr.lines[0].cond

    print ocl(expr)

