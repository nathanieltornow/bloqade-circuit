"""Random jeff programs, built through bloqade.jeff.build.

Every wire and register is tracked linearly by the generator: a consumed
linear value leaves its pool; region ends yield what is carried and consume
the rest. Registers carried across regions and calls are full. Quantum ops
are restricted to basis-preserving gates outside `h`-sandwiches, so the
reference interpreter sees deterministic measurements.
"""

from __future__ import annotations

import math
import random
from dataclasses import field, dataclass

from kirin import ir, types
from kirin.dialects import func

from bloqade.jeff import WireType, QuregType, IntArrayType, FloatArrayType
from bloqade.jeff.types import qureg
from bloqade.jeff.dialects import stmts, kernel

from .build import add, entry, method, switch, for_loop, while_loop
from .jeff_interpreter import INT_MAX, INT_MIN

INT_CONSTS = [
    0,
    1,
    -1,
    2,
    3,
    5,
    7,
    31,
    100,
    -100,
    INT_MIN,
    INT_MAX,
    INT_MIN + 1,
    INT_MAX - 1,
    1 << 30,
    -(1 << 30),
    65535,
    -65536,
    0x55555555,
    -0x55555556,
]
FLOAT_CONSTS = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    math.pi,
    -math.pi,
    2 * math.pi,
    math.pi / 2,
    1e6,
    -1e6,
    1e-6,
    3.75,
    -2.25,
    100.0,
]
LOOP_BOUNDS = [
    (0, 3, 1),
    (3, 0, -1),
    (0, 0, 1),
    (2, -3, -2),
    (-2, 2, 1),
    (5, 1, 1),
    (0, 6, 2),
    (1, 8, 3),
    (INT_MAX - 2, INT_MAX, 1),
    (INT_MIN, INT_MIN + 3, 1),
    (0, 4, 1),
    (-1, 3, 1),
    (4, -1, -1),
    (0, 1, 1),
    (7, 0, -3),
    (0, 5, 7),
]


def output_type(values: list[ir.SSAValue]) -> types.TypeAttribute:
    """Return the output type for returned values: none, one, or a tuple."""
    if not values:
        return types.NoneType
    if len(values) == 1:
        return values[0].type
    return types.Generic(tuple, *(value.type for value in values))


@dataclass(eq=False)
class Reg:
    ssa: ir.SSAValue
    slots: list[bool]  # True = filled
    pending: ir.SSAValue | None = None  # loop index an extract left empty

    @property
    def n(self):
        return len(self.slots)

    @property
    def full(self):
        return all(self.slots)


@dataclass(eq=False)
class Arr:
    ssa: ir.SSAValue
    n: int
    kind: str  # "int" | "bit" | "float"


@dataclass
class HelperSig:
    mt: ir.Method
    params: list  # ("int"|"float"|"bit"|"wire"|"reg"|"iarr"|"barr"|"farr", extra)
    returns: list  # ("hand", param_index) | ("int"|"float"|"bit"|"wire"|"iarr"|..., extra) | ("reg", n)
    name: str


@dataclass(eq=False)
class Env:
    block: ir.Block
    ints: list = field(default_factory=list)
    floats: list = field(default_factory=list)
    bits: list = field(default_factory=list)
    wires: list = field(default_factory=list)
    regs: list = field(default_factory=list)
    arrs: list = field(default_factory=list)  # Arr
    origin: dict = field(
        default_factory=dict
    )  # wire ssa -> (Reg, idx ssa, idx value|None)
    index: tuple | None = None  # (ssa, lo, hi) loop index with static inclusive bounds
    depth: int = 0


class Refuse(Exception):
    pass


class Gen:
    def __init__(self, rng: random.Random, probe: bool = False):
        """`probe` also draws constructs the conversion refuses: `while`,
        multi-controlled gates, a zero step."""
        self.rng = rng
        self.helpers: list[HelperSig] = []
        self.counter = 0
        self.probe = probe

    # -- basic pickers ----------------------------------------------------
    def add(self, env: Env, stmt):
        return add(env.block, stmt)

    def const_int(self, env: Env, value: int | None = None) -> ir.SSAValue:
        if value is None:
            r = self.rng.random()
            if r < 0.6:
                value = self.rng.choice(INT_CONSTS)
            elif r < 0.85:
                value = self.rng.randint(-20, 20)
            else:
                value = self.rng.randint(INT_MIN, INT_MAX)
        v = self.add(env, stmts.ConstInt(value=value)).result
        env.ints.append(v)
        return v

    def const_float(self, env: Env, value: float | None = None) -> ir.SSAValue:
        if value is None:
            r = self.rng.random()
            if r < 0.6:
                value = self.rng.choice(FLOAT_CONSTS)
            elif r < 0.9:
                value = self.rng.uniform(-10, 10)
            else:
                value = self.rng.uniform(-1e5, 1e5)
        v = self.add(env, stmts.ConstFloat(value=value)).result
        env.floats.append(v)
        return v

    def const_bit(self, env: Env, value: bool | None = None) -> ir.SSAValue:
        if value is None:
            value = self.rng.random() < 0.5
        v = self.add(env, stmts.ConstInt(value=int(value), bitwidth=1)).result
        env.bits.append(v)
        return v

    def pick_int(self, env: Env) -> ir.SSAValue:
        if env.ints and self.rng.random() < 0.8:
            return self.rng.choice(env.ints)
        return self.const_int(env)

    def pick_float(self, env: Env) -> ir.SSAValue:
        if env.floats and self.rng.random() < 0.8:
            return self.rng.choice(env.floats)
        return self.const_float(env)

    def pick_bit(self, env: Env) -> ir.SSAValue:
        if env.bits and self.rng.random() < 0.8:
            return self.rng.choice(env.bits)
        return self.const_bit(env)

    def nonzero_int(self, env: Env) -> ir.SSAValue:
        """An int that is never zero: a nonzero constant or `x | 1`."""
        if self.rng.random() < 0.5:
            v = self.rng.choice(
                [c for c in INT_CONSTS if c != 0]
                + [self.rng.randint(1, 9), -self.rng.randint(1, 9)]
            )
            return self.const_int(env, v)
        one = self.const_int(env, 1)
        v = self.add(env, stmts.IntOr(self.pick_int(env), one)).result
        env.ints.append(v)
        return v

    def bounded_int(self, env: Env, mask: int) -> ir.SSAValue:
        """`x & mask` for a power-of-two-minus-one mask: in 0..mask."""
        if self.rng.random() < 0.4:
            return self.const_int(env, self.rng.randint(0, mask))
        m = self.const_int(env, mask)
        v = self.add(env, stmts.IntAnd(self.pick_int(env), m)).result
        env.ints.append(v)
        return v

    def index_in(self, env: Env, n: int) -> tuple[ir.SSAValue, int | None]:
        """An index in 0..n-1: a constant (value known), the loop index when
        its range fits, or `x remU n` (value unknown)."""
        assert n > 0
        r = self.rng.random()
        if (
            env.index is not None
            and 0 <= env.index[1]
            and env.index[2] < n
            and r < 0.35
        ):
            return env.index[0], None
        if r < 0.75:
            k = self.rng.randint(0, n - 1)
            return self.const_int(env, k), k
        nn = self.const_int(env, n)
        low = self.bounded_int(env, INT_MAX)
        v = self.add(env, stmts.IntRemS(low, nn)).result
        env.ints.append(v)
        return v, None

    # -- classical statements ---------------------------------------------
    def int_op(self, env: Env):
        kinds = [
            "add",
            "sub",
            "mul",
            "div_s",
            "rem_s",
            "pow",
            "and",
            "or",
            "xor",
            "min_s",
            "max_s",
            "shl",
            "not",
            "abs",
            "select",
            "eq",
            "lt_s",
            "lte_s",
        ]
        if self.probe:
            kinds += ["div_u", "rem_u", "min_u", "max_u", "shr", "lt_u", "lte_u"]
        kind = self.rng.choice(kinds)
        a = self.pick_int(env)
        if kind in ("div_s", "div_u", "rem_s", "rem_u"):
            b = self.nonzero_int(env)
        elif kind in ("shl", "shr"):
            b = self.bounded_int(env, 63 if self.probe else 31)
        elif kind == "pow":
            b = self.bounded_int(env, 15)
            if self.probe and self.rng.random() < 0.3:
                b = self.const_int(env, -self.rng.randint(1, 3))
        else:
            b = self.pick_int(env)
        cls = {
            "add": stmts.IntAdd,
            "sub": stmts.IntSub,
            "mul": stmts.IntMul,
            "div_s": stmts.IntDivS,
            "div_u": stmts.IntDivU,
            "rem_s": stmts.IntRemS,
            "rem_u": stmts.IntRemU,
            "pow": stmts.IntPow,
            "and": stmts.IntAnd,
            "or": stmts.IntOr,
            "xor": stmts.IntXor,
            "min_s": stmts.IntMinS,
            "min_u": stmts.IntMinU,
            "max_s": stmts.IntMaxS,
            "max_u": stmts.IntMaxU,
            "shl": stmts.IntShl,
            "shr": stmts.IntShr,
        }.get(kind)
        if cls is not None:
            env.ints.append(self.add(env, cls(a, b)).result)
        elif kind == "not":
            env.ints.append(self.add(env, stmts.IntNot(a)).result)
        elif kind == "abs":
            env.ints.append(self.add(env, stmts.IntAbs(a)).result)
        elif kind == "select":
            env.ints.append(
                self.add(env, stmts.IntSelect(self.pick_bit(env), a, b)).result
            )
        else:
            cls = {
                "eq": stmts.IntEq,
                "lt_s": stmts.IntLtS,
                "lte_s": stmts.IntLteS,
                "lt_u": stmts.IntLtU,
                "lte_u": stmts.IntLteU,
            }[kind]
            env.bits.append(self.add(env, cls(a, b)).result)

    def float_op(self, env: Env):
        kind = self.rng.choice(
            [
                "add",
                "sub",
                "mul",
                "pow",
                "atan2",
                "max",
                "min",
                "eq",
                "lt",
                "lte",
                "sqrt",
                "abs",
                "ceil",
                "floor",
                "exp",
                "log",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "asinh",
                "acosh",
                "atanh",
                "is_nan",
                "is_inf",
                "select",
            ]
        )
        a = self.pick_float(env)
        if kind in ("add", "sub", "mul", "atan2", "max", "min", "eq", "lt", "lte"):
            b = self.pick_float(env)
            cls = {
                "add": stmts.FloatAdd,
                "sub": stmts.FloatSub,
                "mul": stmts.FloatMul,
                "atan2": stmts.FloatAtan2,
                "max": stmts.FloatMax,
                "min": stmts.FloatMin,
                "eq": stmts.FloatEq,
                "lt": stmts.FloatLt,
                "lte": stmts.FloatLte,
            }[kind]
            r = self.add(env, cls(a, b)).result
            (env.bits if kind in ("eq", "lt", "lte") else env.floats).append(r)
        elif kind == "pow":
            # base = |a| + 1 (never zero, never negative), exponent small
            if self.probe:
                base, e = a, self.pick_float(env)
            else:
                one = self.const_float(env, 1.0)
                base = self.add(
                    env, stmts.FloatAdd(self.add(env, stmts.FloatAbs(a)).result, one)
                ).result
                e = self.const_float(
                    env, self.rng.choice([0.0, 1.0, 2.0, -1.0, 0.5, -2.0, 3.0])
                )
            env.floats.append(self.add(env, stmts.FloatPow(base, e)).result)
        elif kind == "select":
            env.floats.append(
                self.add(
                    env, stmts.FloatSelect(self.pick_bit(env), a, self.pick_float(env))
                ).result
            )
        elif kind in ("is_nan", "is_inf"):
            cls = stmts.FloatIsNan if kind == "is_nan" else stmts.FloatIsInf
            env.bits.append(self.add(env, cls(a)).result)
        else:
            x = a
            if not self.probe:
                # keep the argument in the function's domain
                if kind in ("sqrt", "log"):
                    x = self.add(env, stmts.FloatAbs(x)).result
                    if kind == "log":
                        x = self.add(
                            env, stmts.FloatAdd(x, self.const_float(env, 1.0))
                        ).result
                elif kind in ("asin", "acos", "atanh"):
                    x = self.add(env, stmts.FloatTanh(x)).result  # in (-1, 1)
                elif kind == "acosh":
                    x = self.add(
                        env,
                        stmts.FloatAdd(
                            self.add(env, stmts.FloatAbs(x)).result,
                            self.const_float(env, 1.0),
                        ),
                    ).result
                elif kind in ("exp", "sinh", "cosh"):
                    hi, lo = self.const_float(env, 50.0), self.const_float(env, -50.0)
                    x = self.add(
                        env,
                        stmts.FloatMin(self.add(env, stmts.FloatMax(x, lo)).result, hi),
                    ).result
            cls = {
                "sqrt": stmts.FloatSqrt,
                "abs": stmts.FloatAbs,
                "ceil": stmts.FloatCeil,
                "floor": stmts.FloatFloor,
                "exp": stmts.FloatExp,
                "log": stmts.FloatLog,
                "sin": stmts.FloatSin,
                "cos": stmts.FloatCos,
                "tan": stmts.FloatTan,
                "asin": stmts.FloatAsin,
                "acos": stmts.FloatAcos,
                "atan": stmts.FloatAtan,
                "sinh": stmts.FloatSinh,
                "cosh": stmts.FloatCosh,
                "tanh": stmts.FloatTanh,
                "asinh": stmts.FloatAsinh,
                "acosh": stmts.FloatAcosh,
                "atanh": stmts.FloatAtanh,
            }[kind]
            env.floats.append(self.add(env, cls(x)).result)

    def bit_op(self, env: Env):
        kind = self.rng.choice(
            ["not", "and", "or", "xor", "eq", "lt_s", "select_int", "add"]
        )
        a = self.pick_bit(env)
        if kind == "not":
            s = self.add(env, stmts.IntNot(a))
            s.result.type = types.Bool
            env.bits.append(s.result)
        elif kind in ("and", "or", "xor", "add"):
            if kind == "add" and not self.probe:
                kind = "xor"
            cls = {
                "and": stmts.IntAnd,
                "or": stmts.IntOr,
                "xor": stmts.IntXor,
                "add": stmts.IntAdd,
            }[kind]
            s = self.add(env, cls(a, self.pick_bit(env)))
            s.result.type = types.Bool
            env.bits.append(s.result)
        elif kind in ("eq", "lt_s"):
            cls = stmts.IntEq if kind == "eq" else stmts.IntLtS
            env.bits.append(self.add(env, cls(a, self.pick_bit(env))).result)
        else:  # bit to int
            env.ints.append(
                self.add(
                    env,
                    stmts.IntSelect(a, self.const_int(env, 1), self.const_int(env, 0)),
                ).result
            )

    def array_op(self, env: Env):
        kind = self.rng.choice(
            ["create", "const", "zero_fill", "get", "len", "set_bad", "get"]
        )
        elem = self.rng.choice(["int", "bit", "float"])
        if kind == "create":
            n = self.rng.randint(1, 4)
            if elem == "int":
                vals = tuple(self.pick_int(env) for _ in range(n))
                s = self.add(env, stmts.IntArrayCreate(vals, bitwidth=32))
            elif elem == "bit":
                vals = tuple(self.pick_bit(env) for _ in range(n))
                s = self.add(env, stmts.IntArrayCreate(vals, bitwidth=1))
            else:
                vals = tuple(self.pick_float(env) for _ in range(n))
                s = self.add(env, stmts.FloatArrayCreate(vals))
            env.arrs.append(Arr(s.result, n, elem))
        elif kind == "const":
            n = self.rng.randint(1, 5)
            if elem == "int":
                vals = tuple(self.rng.choice(INT_CONSTS) for _ in range(n))
                s = self.add(env, stmts.IntArrayConst(values=vals, bitwidth=32))
            elif elem == "bit":
                vals = tuple(self.rng.randint(0, 1) for _ in range(n))
                s = self.add(env, stmts.IntArrayConst(values=vals, bitwidth=1))
            else:
                vals = tuple(self.rng.choice(FLOAT_CONSTS) for _ in range(n))
                s = self.add(env, stmts.FloatArrayConst(values=vals))
            env.arrs.append(Arr(s.result, n, elem))
        elif kind == "zero_fill":
            n = self.rng.randint(1, 4)
            dyn = self.probe and self.rng.random() < 0.3
            size = self.pick_int(env) if dyn else self.const_int(env, n)
            if elem == "float":
                arr = self.add(env, stmts.FloatArrayZero(size)).result
            else:
                arr = self.add(
                    env, stmts.IntArrayZero(size, bitwidth=1 if elem == "bit" else 32)
                ).result
            for _ in range(self.rng.randint(0, 3)):
                k = self.const_int(env, self.rng.randint(0, n - 1))
                if elem == "float":
                    arr = self.add(
                        env, stmts.FloatArraySet(arr, k, self.pick_float(env))
                    ).result
                elif elem == "bit":
                    arr = self.add(
                        env, stmts.IntArraySet(arr, k, self.pick_bit(env))
                    ).result
                else:
                    arr = self.add(
                        env, stmts.IntArraySet(arr, k, self.pick_int(env))
                    ).result
            env.arrs.append(Arr(arr, n, elem))
        elif kind == "get":
            if not env.arrs:
                return self.array_op(env)
            arr = self.rng.choice(env.arrs)
            if arr.n == 0:
                return
            idx, _ = self.index_in(env, arr.n)
            if arr.kind == "float":
                env.floats.append(
                    self.add(env, stmts.FloatArrayGet(arr.ssa, idx)).result
                )
            elif arr.kind == "bit":
                env.bits.append(
                    self.add(env, stmts.IntArrayGet(arr.ssa, idx, bitwidth=1)).result
                )
            else:
                env.ints.append(
                    self.add(env, stmts.IntArrayGet(arr.ssa, idx, bitwidth=32)).result
                )
        elif kind == "len":
            if not env.arrs:
                return self.array_op(env)
            arr = self.rng.choice(env.arrs)
            cls = stmts.FloatArrayLen if arr.kind == "float" else stmts.IntArrayLen
            env.ints.append(self.add(env, cls(arr.ssa)).result)
        elif kind == "set_bad":
            # a set on an array of unknown contents: refused by the conversion
            if not env.arrs or not self.probe:
                return
            arr = self.rng.choice(env.arrs)
            if arr.n == 0:
                return
            idx, _ = self.index_in(env, arr.n)
            if arr.kind == "float":
                r = self.add(
                    env, stmts.FloatArraySet(arr.ssa, idx, self.pick_float(env))
                ).result
            elif arr.kind == "bit":
                r = self.add(
                    env, stmts.IntArraySet(arr.ssa, idx, self.pick_bit(env))
                ).result
            else:
                r = self.add(
                    env, stmts.IntArraySet(arr.ssa, idx, self.pick_int(env))
                ).result
            env.arrs.append(Arr(r, arr.n, arr.kind))

    # -- quantum statements -----------------------------------------------
    def alloc(self, env: Env):
        env.wires.append(self.add(env, stmts.Alloc()).result)

    def reg_alloc(self, env: Env):
        n = self.rng.randint(1, 4)
        r = self.rng.random()
        if r < 0.5:
            s = self.add(env, stmts.RegAlloc(self.const_int(env, n)))
            if self.rng.random() < 0.5:
                s.result.type = qureg(n)
            env.regs.append(Reg(s.result, [True] * n))
        elif r < 0.8 or not env.wires:
            wires = []
            for _ in range(n):
                if env.wires and self.rng.random() < 0.5:
                    w = self.rng.choice(env.wires)
                    self.take_wire(env, w)
                else:
                    w = self.add(env, stmts.Alloc()).result
                wires.append(w)
            s = self.add(env, stmts.RegCreate(tuple(wires)))
            env.regs.append(Reg(s.result, [True] * n))
        else:
            # allocate with a dynamic size: (x & 3) + 1
            k = self.rng.randint(0, 3)
            size = self.add(
                env, stmts.IntAdd(self.const_int(env, k), self.const_int(env, 1))
            ).result
            env.ints.append(size)
            s = self.add(env, stmts.RegAlloc(size))
            env.regs.append(Reg(s.result, [True] * (k + 1)))

    def take_wire(self, env: Env, w):
        env.wires.remove(w)
        env.origin.pop(w, None)

    def rewire(self, env: Env, old, new):
        """`new` continues `old`: same pool slot and origin."""
        env.wires[env.wires.index(old)] = new
        if old in env.origin:
            env.origin[new] = env.origin.pop(old)

    def single_gate(self, env: Env, w, sandwich: bool = False):
        """A basis-preserving single-qubit gate on w; in a sandwich, a gate
        whose h-conjugate is basis-preserving."""
        pi = math.pi
        if sandwich:
            choice = self.rng.choice(
                [
                    "z",
                    "s2",
                    "t4",
                    "rzpi",
                    "x",
                    "y",
                    "u0",
                    "sdg2",
                    "i",
                    "rzpi_adj",
                    "zpow",
                ]
            )
        else:
            choice = self.rng.choice(
                [
                    "x",
                    "y",
                    "z",
                    "s",
                    "t",
                    "sdg",
                    "tdg",
                    "rxpi",
                    "rypi",
                    "rz",
                    "u",
                    "i",
                    "xpow",
                    "x_adj",
                    "custom_x",
                    "hadamard_pair",
                    "r1",
                    "sqrt_x_pair",
                    "gphase",
                ]
            )
        params: tuple = ()
        name, adjoint, power = "x", False, 1
        if choice in ("x", "y", "z", "s", "t", "i"):
            name = choice
        elif choice == "sdg":
            name, adjoint = "s", True
        elif choice == "tdg":
            name, adjoint = "t", True
        elif choice == "x_adj":
            name, adjoint = self.rng.choice(["x", "y", "z"]), True
        elif choice == "xpow":
            name, power = self.rng.choice(["x", "y", "z"]), self.rng.randint(1, 3)
        elif choice == "custom_x":
            name = self.rng.choice(
                ["PauliX", "paulix", "PauliY", "PauliZ", "X", "Y", "Z"]
            )
        elif choice == "rxpi":
            name = "rx"
            params = (
                self.const_float(env, self.rng.choice([-3, -2, -1, 1, 2, 3, 0]) * pi),
            )
            adjoint = self.rng.random() < 0.3
        elif choice == "rypi":
            name = "ry"
            params = (
                self.const_float(env, self.rng.choice([-3, -2, -1, 1, 2, 3, 0]) * pi),
            )
            adjoint = self.rng.random() < 0.3
        elif choice in ("rz", "r1"):
            name = choice
            params = (self.pick_float(env),)
            adjoint = self.rng.random() < 0.3
        elif choice == "gphase":
            if not self.probe:
                name = "z"
            else:
                params = (self.pick_float(env),)
                self.add(env, stmts.Gate((), (), params, gate_name="gphase"))
                return
        elif choice == "u":
            name = "u"
            params = (
                self.const_float(env, self.rng.choice([0.0, pi, -pi, 2 * pi])),
                self.pick_float(env),
                self.pick_float(env),
            )
            adjoint = self.rng.random() < 0.3
        elif choice == "hadamard_pair":
            for nm in (self.rng.choice(["h", "Hadamard", "H"]), "h"):
                g = self.add(env, stmts.Gate((w,), (), (), gate_name=nm))
                self.rewire(env, w, g.results[0])
                w = g.results[0]
            return
        elif choice == "sqrt_x_pair":
            nm = self.rng.choice(["sqrt_x", "sqrt_y", "sx", "SX"])
            g = self.add(env, stmts.Gate((w,), (), (), gate_name=nm))
            self.rewire(env, w, g.results[0])
            w = g.results[0]
            g = self.add(env, stmts.Gate((w,), (), (), gate_name=nm))
            self.rewire(env, w, g.results[0])
            return
        # sandwich choices
        elif choice == "s2":
            name, power = "s", 2
        elif choice == "sdg2":
            name, power, adjoint = "s", 2, True
        elif choice == "t4":
            name, power = "t", 4
        elif choice == "zpow":
            name, power = "z", self.rng.randint(1, 3)
        elif choice == "rzpi":
            name, params = (
                "rz",
                (self.const_float(env, self.rng.choice([pi, -pi, 3 * pi])),),
            )
        elif choice == "rzpi_adj":
            name, params, adjoint = "rz", (self.const_float(env, pi),), True
        elif choice == "u0":
            phi = self.rng.uniform(-3, 3)
            name = "u"
            params = (
                self.const_float(env, 0.0),
                self.const_float(env, phi),
                self.const_float(env, pi - phi),
            )
        g = self.add(
            env,
            stmts.Gate((w,), (), params, gate_name=name, adjoint=adjoint, power=power),
        )
        self.rewire(env, w, g.results[0])

    def multi_gate(self, env: Env):
        if len(env.wires) < 2:
            return self.alloc(env)
        choice = self.rng.choice(
            [
                "cx",
                "cx_custom",
                "cz",
                "swap",
                "cy",
                "ccz",
                "ccx",
                "cnot3",
                "crx",
                "cswap",
                "ccz_custom",
            ]
        )
        if choice in ("ccz", "ccx", "cnot3", "ccz_custom") and len(env.wires) < 3:
            choice = "cx"
        if choice in ("ccx", "cnot3", "crx", "cswap") and not self.probe:
            choice = "cz"
        ws = self.rng.sample(
            env.wires,
            3 if choice in ("ccz", "ccx", "cnot3", "ccz_custom", "cswap") else 2,
        )
        params: tuple = ()
        if choice == "cx":
            g = self.add(env, stmts.Gate((ws[1],), (ws[0],), (), gate_name="x"))
        elif choice == "cy":
            g = self.add(env, stmts.Gate((ws[1],), (ws[0],), (), gate_name="y"))
        elif choice == "cx_custom":
            g = self.add(
                env,
                stmts.Gate(
                    (ws[0], ws[1]),
                    (),
                    (),
                    gate_name=self.rng.choice(["cx", "CX", "cnot", "CNOT"]),
                ),
            )
        elif choice == "cz":
            if self.rng.random() < 0.5:
                g = self.add(env, stmts.Gate((ws[1],), (ws[0],), (), gate_name="z"))
            else:
                g = self.add(env, stmts.Gate((ws[0], ws[1]), (), (), gate_name="cz"))
        elif choice == "swap":
            g = self.add(
                env,
                stmts.Gate(
                    (ws[0], ws[1]),
                    (),
                    (),
                    gate_name="swap",
                    power=self.rng.choice([1, 1, 2, 3]),
                ),
            )
        elif choice == "ccz":
            g = self.add(env, stmts.Gate((ws[2],), (ws[0], ws[1]), (), gate_name="z"))
        elif choice == "ccz_custom":
            g = self.add(
                env,
                stmts.Gate(
                    (ws[0], ws[1], ws[2]),
                    (),
                    (),
                    gate_name=self.rng.choice(["ccz", "CCZ"]),
                ),
            )
        elif choice == "ccx":
            g = self.add(env, stmts.Gate((ws[2],), (ws[0], ws[1]), (), gate_name="x"))
        elif choice == "cnot3":
            g = self.add(
                env,
                stmts.Gate(
                    (ws[0], ws[1], ws[2]),
                    (),
                    (),
                    gate_name=self.rng.choice(["ccx", "toffoli"]),
                ),
            )
        elif choice == "crx":
            params = (self.const_float(env, math.pi),)
            g = self.add(env, stmts.Gate((ws[1],), (ws[0],), params, gate_name="rx"))
        else:  # cswap
            g = self.add(
                env, stmts.Gate((ws[1], ws[2]), (ws[0],), (), gate_name="swap")
            )
        for old, new in zip(ws, g.results):
            self.rewire(env, old, new)

    def sandwich(self, env: Env):
        """h(q); gates with q as target only; h(q)."""
        if not env.wires:
            self.alloc(env)
        q = self.rng.choice(env.wires)
        g = self.add(
            env, stmts.Gate((q,), (), (), gate_name=self.rng.choice(["h", "Hadamard"]))
        )
        self.rewire(env, q, g.results[0])
        q = g.results[0]
        for _ in range(self.rng.randint(1, 3)):
            others = [w for w in env.wires if w is not q]
            r = self.rng.random()
            if others and r < 0.3:
                c = self.rng.choice(others)
                g = self.add(env, stmts.Gate((q,), (c,), (), gate_name="z"))
                self.rewire(env, q, g.results[0])
                self.rewire(env, c, g.results[1])
                q = g.results[0]
            elif len(others) >= 2 and r < 0.5:
                c1, c2 = self.rng.sample(others, 2)
                g = self.add(env, stmts.Gate((q,), (c1, c2), (), gate_name="z"))
                self.rewire(env, q, g.results[0])
                self.rewire(env, c1, g.results[1])
                self.rewire(env, c2, g.results[2])
                q = g.results[0]
            elif others and r < 0.6:
                c = self.rng.choice(others)
                g = self.add(env, stmts.Gate((q,), (c,), (), gate_name="x"))
                self.rewire(env, q, g.results[0])
                self.rewire(env, c, g.results[1])
                q = g.results[0]
            else:
                pos = env.wires.index(q)
                self.single_gate(env, q, sandwich=True)
                q = env.wires[pos]
        g = self.add(env, stmts.Gate((q,), (), (), gate_name="h"))
        self.rewire(env, q, g.results[0])

    def measure(self, env: Env):
        if not env.wires:
            return self.alloc(env)
        w = self.rng.choice(env.wires)
        kind = self.rng.choice(
            ["nd", "nd", "destructive", "reset", "free", "reset_free_zero"]
        )
        if kind == "nd":
            m = self.add(env, stmts.MeasureNd(w))
            self.rewire(env, w, m.result_wire)
            env.bits.append(m.bit)
        elif kind == "destructive":
            m = self.add(env, stmts.Measure(w))
            self.take_wire(env, w)
            env.bits.append(m.bit)
        elif kind == "reset":
            r = self.add(env, stmts.Reset(w))
            self.rewire(env, w, r.result)
        elif kind == "free":
            self.add(env, stmts.Free(w))
            self.take_wire(env, w)
        else:
            r = self.add(env, stmts.Reset(w))
            self.add(env, stmts.FreeZero(r.result))
            self.take_wire(env, w)

    def reg_op(self, env: Env):
        if not env.regs:
            return self.reg_alloc(env)
        reg = self.rng.choice(env.regs)
        kind = self.rng.choice(
            [
                "extract",
                "insert",
                "extract_slice",
                "insert_slice",
                "split",
                "join",
                "length",
                "free",
                "gate_elem",
            ]
        )
        if kind == "extract" or kind == "gate_elem":
            filled = [i for i, f in enumerate(reg.slots) if f]
            if not filled:
                return
            if (
                env.index is not None
                and 0 <= env.index[1]
                and env.index[2] < reg.n
                and reg.full
                and self.rng.random() < 0.5
            ):
                # loop-index extract: the slot must be refilled at the same index before the yield
                idx, k = env.index[0], None
            else:
                k = self.rng.choice(filled)
                idx = self.const_int(env, k)
            e = self.add(env, stmts.Extract(reg.ssa, idx))
            reg.ssa = e.result_reg
            if k is not None:
                reg.slots[k] = False
            else:
                reg.slots = [None] * reg.n  # unknown until reinserted at idx
                reg.pending = idx
            env.wires.append(e.wire)
            env.origin[e.wire] = (reg, idx, k)
            if kind == "gate_elem":
                self.single_gate(env, e.wire)
        elif kind == "insert":
            self.insert_something(env, reg)
        elif kind == "extract_slice":
            if not reg.full or reg.n == 0:
                return
            n = self.rng.randint(1, reg.n)
            a = self.rng.randint(0, reg.n - n)
            e = self.add(
                env,
                stmts.ExtractSlice(
                    reg.ssa, self.const_int(env, a), self.const_int(env, n)
                ),
            )
            reg.ssa = e.result_reg
            for i in range(a, a + n):
                reg.slots[i] = False
            part = Reg(e.slice_reg, [True] * n)
            env.regs.append(part)
            env.origin[e.slice_reg] = (reg, e.start, a)
        elif kind == "insert_slice":
            # a full register whose length fits a run of empty slots
            holes = [
                r for r in env.regs if r.slots and any(f is False for f in r.slots)
            ]
            if not holes:
                return
            target = self.rng.choice(holes)
            cands = [
                r
                for r in env.regs
                if r is not target and r.full and 0 < r.n <= target.n
            ]
            if not cands:
                return
            part = self.rng.choice(cands)
            runs = [
                a
                for a in range(target.n - part.n + 1)
                if all(target.slots[i] is False for i in range(a, a + part.n))
            ]
            if not runs:
                return
            a = self.rng.choice(runs)
            origin = env.origin.get(part.ssa)
            if (
                origin is not None
                and origin[0] is target
                and origin[2] == a
                and self.rng.random() < 0.5
            ):
                start = origin[1]
            else:
                start = self.const_int(env, a)
            s = self.add(env, stmts.InsertSlice(target.ssa, start, part.ssa))
            env.regs.remove(part)
            env.origin.pop(part.ssa, None)
            target.ssa = s.result
            for i in range(a, a + part.n):
                target.slots[i] = True
        elif kind == "split":
            if any(f is None for f in reg.slots):
                return
            k = self.rng.randint(0, reg.n)
            s = self.add(env, stmts.RegSplit(reg.ssa, self.const_int(env, k)))
            env.regs.remove(reg)
            self.forget_origin(env, reg)
            env.regs.append(Reg(s.before, reg.slots[:k]))
            env.regs.append(Reg(s.after, reg.slots[k:]))
        elif kind == "join":
            others = [
                r
                for r in env.regs
                if r is not reg and all(f is not None for f in r.slots)
            ]
            if not others or any(f is None for f in reg.slots):
                return
            other = self.rng.choice(others)
            s = self.add(env, stmts.RegJoin(reg.ssa, other.ssa))
            env.regs.remove(reg)
            env.regs.remove(other)
            self.forget_origin(env, reg)
            self.forget_origin(env, other)
            env.regs.append(Reg(s.result, reg.slots + other.slots))
        elif kind == "length":
            s = self.add(env, stmts.RegLength(reg.ssa))
            reg.ssa = s.result_reg
            env.ints.append(s.length)
        elif kind == "free":
            if any(f is None for f in reg.slots):
                return
            if self.rng.random() < 0.2 and reg.full:
                # reset every slot then free_zero: walk the slots
                for i in range(reg.n):
                    e = self.add(env, stmts.Extract(reg.ssa, self.const_int(env, i)))
                    r = self.add(env, stmts.Reset(e.wire))
                    ins = self.add(
                        env,
                        stmts.Insert(e.result_reg, self.const_int(env, i), r.result),
                    )
                    reg.ssa = ins.result
                self.add(env, stmts.RegFreeZero(reg.ssa))
            else:
                self.add(env, stmts.RegFree(reg.ssa))
            env.regs.remove(reg)
            self.forget_origin(env, reg)

    def forget_origin(self, env: Env, reg: Reg):
        for w, (r, _, _) in list(env.origin.items()):
            if r is reg:
                del env.origin[w]

    def insert_something(self, env: Env, reg: Reg):
        """Fill one empty slot of `reg`: with its own wire back (same or an
        equal constant index), with another wire, or with a fresh one."""
        if any(f is None for f in reg.slots):
            # loop-index extracted: must insert at the loop index
            own = [
                w
                for w, (r, idx, k) in env.origin.items()
                if r is reg and k is None and w in env.wires
            ]
            idx = reg.pending
            r = self.rng.random()
            if own and r < 0.6:
                w = own[0]
            elif env.wires and r < 0.8:
                w = self.rng.choice(env.wires)
            else:
                w = self.add(env, stmts.Alloc()).result
                env.wires.append(w)
            s = self.add(env, stmts.Insert(reg.ssa, idx, w))
            self.take_wire(env, w)
            reg.ssa = s.result
            reg.slots = [True] * reg.n
            reg.pending = None
            self.forget_origin(env, reg)
            return
        empty = [i for i, f in enumerate(reg.slots) if not f]
        if not empty:
            return
        k = self.rng.choice(empty)
        own = [
            w
            for w, (r, idx, kk) in env.origin.items()
            if r is reg and kk == k and w in env.wires
        ]
        r = self.rng.random()
        if own and r < 0.5:
            w = own[0]
            idx = (
                env.origin[w][1] if self.rng.random() < 0.5 else self.const_int(env, k)
            )
        elif env.wires and r < 0.8:
            w = self.rng.choice(env.wires)
            idx = self.const_int(env, k)
        else:
            w = self.add(env, stmts.Alloc()).result
            env.wires.append(w)
            idx = self.const_int(env, k)
        s = self.add(env, stmts.Insert(reg.ssa, idx, w))
        self.take_wire(env, w)
        reg.ssa = s.result
        reg.slots[k] = True

    def fill_reg(self, env: Env, reg: Reg):
        """Make `reg` full before it crosses a region or call boundary."""
        guard = 0
        while not reg.full and guard < 20:
            guard += 1
            self.insert_something(env, reg)
        assert reg.full, reg.slots

    # -- control flow -----------------------------------------------------
    def carried(self, env: Env, allow_linear=True):
        """A random selection of values to carry: (kinds, values)."""
        kinds, values = [], []
        for pool, kind in (
            (env.ints, "int"),
            (env.floats, "float"),
            (env.bits, "bit"),
            (env.arrs, "arr"),
        ):
            for v in pool:
                if self.rng.random() < 0.3:
                    kinds.append((kind, v if kind == "arr" else None))
                    values.append(v.ssa if kind == "arr" else v)
        if allow_linear:
            for w in list(env.wires):
                if self.rng.random() < 0.5:
                    kinds.append(("wire", None))
                    values.append(w)
                    self.take_wire(env, w)
            for reg in list(env.regs):
                if all(f is not None for f in reg.slots) and self.rng.random() < 0.5:
                    self.fill_reg(env, reg)
                    kinds.append(("reg", reg.n))
                    values.append(reg.ssa)
                    env.regs.remove(reg)
                    self.forget_origin(env, reg)
        return kinds, values

    def region_env(self, block: ir.Block, args, kinds, depth) -> Env:
        env = Env(block=block, depth=depth)
        for (kind, extra), a in zip(kinds, args):
            self.bind(env, kind, extra, a)
        return env

    def bind(self, env: Env, kind, extra, a):
        if kind == "int":
            env.ints.append(a)
        elif kind == "float":
            env.floats.append(a)
        elif kind == "bit":
            env.bits.append(a)
        elif kind == "arr":
            env.arrs.append(Arr(a, extra.n, extra.kind))
        elif kind in ("iarr", "barr", "farr"):
            env.arrs.append(
                Arr(a, extra, {"iarr": "int", "barr": "bit", "farr": "float"}[kind])
            )
        elif kind == "wire":
            env.wires.append(a)
        elif kind == "reg":
            env.regs.append(Reg(a, [True] * extra))

    def yields(self, env: Env, kinds) -> list:
        """Values of the given kinds from the env, consuming all linear values."""
        out = []
        for kind, extra in kinds:
            if kind == "int":
                out.append(self.pick_int(env))
            elif kind == "float":
                out.append(self.pick_float(env))
            elif kind == "bit":
                out.append(self.pick_bit(env))
            elif kind == "arr":
                cands = [a for a in env.arrs if a.kind == extra.kind and a.n == extra.n]
                if cands:
                    out.append(self.rng.choice(cands).ssa)
                else:
                    out.append(self.make_arr(env, extra.kind, extra.n).ssa)
            elif kind in ("iarr", "barr", "farr"):
                k = {"iarr": "int", "barr": "bit", "farr": "float"}[kind]
                cands = [a for a in env.arrs if a.kind == k and a.n == extra]
                out.append(
                    (
                        self.rng.choice(cands)
                        if cands
                        else self.make_arr(env, k, extra)
                    ).ssa
                )
            elif kind == "wire":
                if not env.wires:
                    self.alloc(env)
                w = self.rng.choice(env.wires)
                self.take_wire(env, w)
                out.append(w)
            elif kind == "reg":
                cands = [
                    r
                    for r in env.regs
                    if r.n == extra and all(f is not None for f in r.slots)
                ]
                if cands:
                    reg = self.rng.choice(cands)
                    self.fill_reg(env, reg)
                else:
                    s = self.add(env, stmts.RegAlloc(self.const_int(env, extra)))
                    reg = Reg(s.result, [True] * extra)
                    env.regs.append(reg)
                env.regs.remove(reg)
                self.forget_origin(env, reg)
                out.append(reg.ssa)
        self.consume_rest(env)
        return out

    def make_arr(self, env: Env, kind: str, n: int) -> Arr:
        if kind == "int":
            s = self.add(
                env,
                stmts.IntArrayConst(
                    values=tuple(self.rng.choice(INT_CONSTS) for _ in range(n)),
                    bitwidth=32,
                ),
            )
        elif kind == "bit":
            s = self.add(
                env,
                stmts.IntArrayConst(
                    values=tuple(self.rng.randint(0, 1) for _ in range(n)), bitwidth=1
                ),
            )
        else:
            s = self.add(
                env,
                stmts.FloatArrayConst(
                    values=tuple(self.rng.choice(FLOAT_CONSTS) for _ in range(n))
                ),
            )
        arr = Arr(s.result, n, kind)
        env.arrs.append(arr)
        return arr

    def consume_rest(self, env: Env):
        # regs with unknown slots (loop-index extracts) must get their wire back first
        for reg in list(env.regs):
            if any(f is None for f in reg.slots):
                self.insert_something(env, reg)
                if any(f is None for f in reg.slots):
                    raise Refuse("could not refill loop-indexed register")
        for w in list(env.wires):
            r = self.rng.random()
            if r < 0.4:
                self.add(env, stmts.Free(w))
            elif r < 0.7:
                m = self.add(env, stmts.Measure(w))
                env.bits.append(m.bit)
            else:
                rs = self.add(env, stmts.Reset(w))
                self.add(env, stmts.FreeZero(rs.result))
            self.take_wire(env, w)
        for reg in list(env.regs):
            self.add(env, stmts.RegFree(reg.ssa))
            env.regs.remove(reg)
        env.origin.clear()

    def for_stmt(self, env: Env):
        kinds, values = self.carried(env)
        r = self.rng.random()
        index_bounds = None
        reg_n = [n for (k, n) in kinds if k == "reg"]
        if r < 0.25 and reg_n:
            n = self.rng.choice(reg_n)
            if self.rng.random() < 0.5:
                start, stop, step = (
                    self.const_int(env, 0),
                    self.const_int(env, n),
                    self.const_int(env, 1),
                )
            else:
                start, stop, step = (
                    self.const_int(env, n - 1),
                    self.const_int(env, -1),
                    self.const_int(env, -1),
                )
            index_bounds = (0, n - 1)
        elif r < 0.5:
            start = self.const_int(env, 0)
            stop = self.bounded_int(env, 3)
            step = self.const_int(env, self.rng.choice([1, 1, 2]))
            index_bounds = (0, 3)
        else:
            a, b, c = self.rng.choice(LOOP_BOUNDS)
            start, stop, step = (
                self.const_int(env, a),
                self.const_int(env, b),
                self.const_int(env, c),
            )
            its = list(range(a, b, c))
            index_bounds = (min(its), max(its)) if its else (0, -1)
            if self.probe and self.rng.random() < 0.1:
                step = self.const_int(env, 0)
        depth = env.depth + 1

        def fill(body, i, *args):
            benv = self.region_env(body, args, kinds, depth)
            benv.ints.append(i)
            benv.index = (i, *index_bounds)
            self.fill_block(benv, self.rng.randint(1, 5))
            return self.yields(benv, kinds)

        loop = for_loop(env.block, start, stop, step, values, fill)
        for (kind, extra), res in zip(kinds, loop.results):
            self.bind(env, kind, extra, res)

    def switch_stmt(self, env: Env):
        kinds, values = self.carried(env)
        extra_kinds = [
            (self.rng.choice(["int", "float", "bit"]), None)
            for _ in range(self.rng.randint(0, 2))
        ]
        out_kinds = kinds + extra_kinds
        ncases = self.rng.randint(0, 3)
        r = self.rng.random()
        if r < 0.4:
            sel = self.pick_bit(env)
        elif r < 0.7 and ncases > 0:
            sel = self.bounded_int(env, ncases if (ncases & (ncases + 1)) == 0 else 3)
        else:
            sel = self.pick_int(env)
        depth = env.depth + 1

        def branch(body, *args):
            benv = self.region_env(body, args, kinds, depth)
            self.fill_block(benv, self.rng.randint(0, 4))
            return self.yields(benv, out_kinds)

        sw = switch(env.block, sel, values, [branch] * ncases, branch)
        for (kind, extra), res in zip(out_kinds, sw.results):
            self.bind(env, kind, extra, res)

    def while_stmt(self, env: Env):
        # refused by the conversion; a bounded counter loop
        kinds, values = self.carried(env, allow_linear=False)
        n = self.const_int(env, self.rng.randint(0, 3))
        kinds = [("int", None)] + kinds
        values = [n] + values

        def before(body, *args):
            benv = self.region_env(body, args, kinds, env.depth + 1)
            zero = self.const_int(benv, 0)
            cond = self.add(benv, stmts.IntLtS(zero, args[0])).result
            return [cond, *args]

        def after(body, *args):
            benv = self.region_env(body, args, kinds, env.depth + 1)
            one = self.const_int(benv, 1)
            dec = self.add(benv, stmts.IntSub(args[0], one)).result
            return [dec, *args[1:]]

        w = while_loop(env.block, values, before, after)
        for (kind, extra), res in zip(kinds, w.results):
            self.bind(env, kind, extra, res)

    # -- calls ------------------------------------------------------------
    def call_stmt(self, env: Env):
        if not self.helpers:
            return
        h = self.rng.choice(self.helpers)
        inputs = []
        for kind, extra in h.params:
            if kind == "int":
                inputs.append(self.pick_int(env))
            elif kind == "float":
                inputs.append(self.pick_float(env))
            elif kind == "bit":
                inputs.append(self.pick_bit(env))
            elif kind in ("iarr", "barr", "farr"):
                k = {"iarr": "int", "barr": "bit", "farr": "float"}[kind]
                cands = [a for a in env.arrs if a.kind == k and a.n == extra]
                inputs.append(
                    (
                        self.rng.choice(cands)
                        if cands
                        else self.make_arr(env, k, extra)
                    ).ssa
                )
            elif kind == "wire":
                if not env.wires:
                    self.alloc(env)
                w = self.rng.choice(env.wires)
                self.take_wire(env, w)
                inputs.append(w)
            elif kind == "reg":
                cands = [
                    r
                    for r in env.regs
                    if r.n == extra and all(f is not None for f in r.slots)
                ]
                if cands:
                    reg = self.rng.choice(cands)
                    self.fill_reg(env, reg)
                else:
                    s = self.add(env, stmts.RegAlloc(self.const_int(env, extra)))
                    reg = Reg(s.result, [True] * extra)
                    env.regs.append(reg)
                env.regs.remove(reg)
                self.forget_origin(env, reg)
                inputs.append(reg.ssa)
        result_types = []
        for kind, extra in h.returns:
            if kind == "hand":
                pk, pe = h.params[extra]
                result_types.append(WireType if pk == "wire" else QuregType)
            else:
                result_types.append(
                    {
                        "int": types.Int,
                        "float": types.Float,
                        "bit": types.Bool,
                        "wire": WireType,
                        "reg": QuregType,
                        "iarr": IntArrayType,
                        "barr": IntArrayType,
                        "farr": FloatArrayType,
                    }[kind]
                )
        call = self.add(env, stmts.Call(h.mt, tuple(inputs), result_types))
        for (kind, extra), res in zip(h.returns, call.results):
            if kind == "hand":
                pk, pe = h.params[extra]
                self.bind(env, pk, pe, res)
            else:
                self.bind(env, kind, extra, res)

    def make_helper(self, index: int):
        name = f"helper{index}_{self.counter}"
        self.counter += 1
        params = []
        for _ in range(self.rng.randint(1, 4)):
            kind = self.rng.choice(
                [
                    "int",
                    "float",
                    "bit",
                    "wire",
                    "wire",
                    "reg",
                    "reg",
                    "iarr",
                    "barr",
                    "farr",
                ]
            )
            extra = None
            if kind == "reg" or kind in ("iarr", "barr", "farr"):
                extra = self.rng.randint(1, 3)
            params.append((kind, extra))
        in_types = tuple(
            {
                "int": types.Int,
                "float": types.Float,
                "bit": types.Bool,
                "wire": WireType,
                "reg": QuregType,
                "iarr": IntArrayType,
                "barr": IntArrayType,
                "farr": FloatArrayType,
            }[k]
            for k, _ in params
        )
        block, args = entry(*in_types)
        env = Env(block=block, depth=0)
        for (kind, extra), a in zip(params, args):
            self.bind(env, kind, extra, a)
        self.fill_block(env, self.rng.randint(2, 8))
        # decide what is handed back: a param whose *value* is still in the pool, or
        # any linear value of the right kind (different identity: still a "hand" in the
        # generator's bookkeeping, but not a continuation for the conversion)
        hand = []
        for i, ((kind, extra), a) in enumerate(zip(params, args)):
            if kind in ("wire", "reg") and self.rng.random() < 0.8:
                hand.append(("hand", i))
        others = []
        for _ in range(self.rng.randint(0, 3)):
            others.append(
                (
                    self.rng.choice(
                        ["int", "float", "bit", "iarr", "barr", "farr", "wire", "reg"]
                    ),
                    None,
                )
            )
        # fresh linear outputs
        returns = list(hand)
        for o in others:
            pos = self.rng.randint(0, len(returns))
            returns.insert(pos, o)
        if len(hand) >= 2 and self.rng.random() < 0.3:
            # out of order hand-back: the caller binds each output to its wire
            i, j = [returns.index(h) for h in hand[:2]]
            returns[i], returns[j] = returns[j], returns[i]
        ret_values = []
        out_specs = []
        for kind, extra in returns:
            if kind == "hand":
                pk, pe = params[extra]
                if pk == "wire":
                    if self.rng.random() < 0.7 and env.wires:
                        w = self.rng.choice(
                            env.wires
                        )  # may or may not be the param's chain
                    else:
                        w = self.add(env, stmts.Alloc()).result
                        env.wires.append(w)
                    self.take_wire(env, w)
                    ret_values.append(w)
                else:
                    cands = [
                        r
                        for r in env.regs
                        if r.n == pe and all(f is not None for f in r.slots)
                    ]
                    if cands:
                        reg = self.rng.choice(cands)
                        self.fill_reg(env, reg)
                    else:
                        s = self.add(env, stmts.RegAlloc(self.const_int(env, pe)))
                        reg = Reg(s.result, [True] * pe)
                        env.regs.append(reg)
                    env.regs.remove(reg)
                    self.forget_origin(env, reg)
                    ret_values.append(reg.ssa)
                out_specs.append(("hand", extra))
            elif kind == "wire":
                w = self.add(env, stmts.Alloc()).result
                if self.rng.random() < 0.5:
                    g = self.add(env, stmts.Gate((w,), (), (), gate_name="x"))
                    w = g.results[0]
                ret_values.append(w)
                out_specs.append(("wire", None))
            elif kind == "reg":
                n = self.rng.randint(1, 3)
                s = self.add(env, stmts.RegAlloc(self.const_int(env, n)))
                ret_values.append(s.result)
                out_specs.append(("reg", n))
            else:
                n = self.rng.randint(1, 3)
                if kind in ("iarr", "barr", "farr"):
                    k = {"iarr": "int", "barr": "bit", "farr": "float"}[kind]
                    cands = [a for a in env.arrs if a.kind == k]
                    arr = self.rng.choice(cands) if cands else self.make_arr(env, k, n)
                    ret_values.append(arr.ssa)
                    out_specs.append((kind, arr.n))
                else:
                    ret_values.append(
                        {
                            "int": self.pick_int,
                            "float": self.pick_float,
                            "bit": self.pick_bit,
                        }[kind](env)
                    )
                    out_specs.append((kind, None))
        self.consume_rest(env)
        out = output_type(ret_values)
        mt = method(block, ret_values, out, inputs=in_types, name=name)
        self.helpers.append(HelperSig(mt, params, out_specs, name))

    def make_recursive_helper(self):
        """rec(n, w) -> (w', count): flips w n times through recursion."""
        name = f"rec_{self.counter}"
        self.counter += 1
        block, (n0, w) = entry(types.Int, WireType)
        three = add(block, stmts.ConstInt(value=3)).result
        n = add(block, stmts.IntAnd(n0, three)).result  # bounded depth
        zero = add(block, stmts.ConstInt(value=0)).result
        done = add(block, stmts.IntLteS(n, zero)).result
        holder: list = []

        def base(body, n_, w_):
            z = add(body, stmts.ConstInt(value=0)).result
            return [w_, z]

        def step(body, n_, w_):
            one = add(body, stmts.ConstInt(value=1)).result
            dec = add(body, stmts.IntSub(n_, one)).result
            g = add(body, stmts.Gate((w_,), (), (), gate_name="x"))
            call = add(
                body, stmts.Call(holder[0], (dec, g.results[0]), (WireType, types.Int))
            )
            inc = add(body, stmts.IntAdd(call.results[1], one)).result
            return [call.results[0], inc]

        # selector: done (bit): case 0 = not done -> step, default (1) -> base.
        # The Method object must exist before the body references it.
        fn = func.Function(
            sym_name=name,
            body=ir.Region(block),
            signature=func.Signature(
                inputs=(types.Int, WireType),
                output=types.Generic(tuple, WireType, types.Int),
            ),
        )
        mt = ir.Method(dialects=kernel, code=fn, sym_name=name)
        holder.append(mt)
        sw = switch(block, done, [n, w], [step], base)
        block.stmts.append(stmts.Return(sw.results[0], sw.results[1]))
        self.helpers.append(
            HelperSig(
                mt, [("int", None), ("wire", None)], [("hand", 1), ("int", None)], name
            )
        )

    # -- blocks -----------------------------------------------------------
    def fill_block(self, env: Env, budget: int):
        for _ in range(budget):
            r = self.rng.random()
            if r < 0.14:
                self.int_op(env)
            elif r < 0.24:
                self.float_op(env)
            elif r < 0.30:
                self.bit_op(env)
            elif r < 0.38:
                self.array_op(env)
            elif r < 0.44:
                self.alloc(env)
            elif r < 0.50:
                self.reg_alloc(env)
            elif r < 0.58:
                if env.wires:
                    self.single_gate(env, self.rng.choice(env.wires))
                else:
                    self.alloc(env)
            elif r < 0.64:
                self.multi_gate(env)
            elif r < 0.68:
                self.sandwich(env)
            elif r < 0.76:
                self.measure(env)
            elif r < 0.86:
                self.reg_op(env)
            elif r < 0.90 and env.depth < 2:
                self.for_stmt(env)
            elif r < 0.94 and env.depth < 2:
                self.switch_stmt(env)
            elif r < 0.945 and self.probe and env.depth < 1:
                self.while_stmt(env)
            else:
                self.call_stmt(env)
        # give registers extracted at a loop index their wire back
        for reg in list(env.regs):
            if any(f is None for f in reg.slots):
                self.insert_something(env, reg)

    def program(self):
        self.helpers = []
        nh = self.rng.randint(0, 3)
        for i in range(nh):
            self.make_helper(i)
        if self.rng.random() < 0.15:
            self.make_recursive_helper()
        # entry params
        nparams = self.rng.randint(0, 2)
        pkinds = [self.rng.choice(["int", "float"]) for _ in range(nparams)]
        in_types = tuple(types.Int if k == "int" else types.Float for k in pkinds)
        block, args = entry(*in_types)
        env = Env(block=block, depth=0)
        argvals = []
        for k, a in zip(pkinds, args):
            if k == "int":
                env.ints.append(a)
                argvals.append(
                    self.rng.choice(INT_CONSTS + [self.rng.randint(-50, 50)])
                )
            else:
                env.floats.append(a)
                argvals.append(
                    self.rng.choice(FLOAT_CONSTS + [self.rng.uniform(-5, 5)])
                )
        self.fill_block(env, self.rng.randint(4, 14))
        # entry returns classical values only
        rets = []
        for _ in range(self.rng.randint(1, 4)):
            kind = self.rng.choice(["int", "float", "bit", "arr"])
            if kind == "int":
                rets.append(self.pick_int(env))
            elif kind == "float":
                rets.append(self.pick_float(env))
            elif kind == "bit":
                rets.append(self.pick_bit(env))
            else:
                if env.arrs:
                    rets.append(self.rng.choice(env.arrs).ssa)
                else:
                    rets.append(self.pick_int(env))
        self.consume_rest(env)
        mt = method(block, rets, output_type(rets), inputs=in_types, name="main")
        return mt, tuple(argvals)
