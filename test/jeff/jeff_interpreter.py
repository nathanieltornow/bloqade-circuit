"""Reference interpreter for jeff dialect IR, written from the capnp schema.

Integers are 32-bit two's complement patterns (bitwidth 1 = bits, stored as
bool). Floats are IEEE doubles. Qubits live in a dense state vector; a
measurement, reset or free of a qubit whose marginal is not (numerically)
deterministic raises `Nondeterministic`, so programs that leave the
computational basis at an observation point are discarded rather than
compared. Undefined behaviour the schema names (step 0, empty-slot
extract, ...) and behaviour it leaves open (division by zero, negative
exponents, shifts by >= bitwidth) raise `Undefined`.
"""

from __future__ import annotations

import math
import cmath

import numpy as np
from kirin import ir, types

from bloqade.jeff.gates import canonical
from bloqade.jeff.dialects import stmts

BITS = 32
MASK = (1 << BITS) - 1
INT_MIN = -(1 << (BITS - 1))
INT_MAX = (1 << (BITS - 1)) - 1


class Undefined(Exception):
    """Behaviour the schema does not define (or calls undefined)."""


class Nondeterministic(Exception):
    """A measurement/reset/free on a qubit not in a basis state."""


class Overflow(Exception):
    """An integer result left the int32 range, which squin integers do not wrap."""


class Nonfinite(Exception):
    """A float became inf/nan; IEEE special cases are out of scope."""


def wrap(x: int) -> int:
    if not INT_MIN <= x <= INT_MAX:
        raise Overflow(f"{x} leaves the int32 range")
    x &= MASK
    return x - (1 << BITS) if x >= 1 << (BITS - 1) else x


def pattern(x: int) -> int:
    return x & MASK


def is_bit(v: ir.SSAValue) -> bool:
    return v.type.is_subseteq(types.Bool)


# -- quantum state ----------------------------------------------------------


class QState:
    def __init__(self):
        self.ids: list[int] = []  # qubit id per axis
        self.psi = np.ones((), dtype=complex)
        self.next_id = 0

    def alloc(self) -> int:
        qid = self.next_id
        self.next_id += 1
        self.ids.append(qid)
        self.psi = np.stack([self.psi, np.zeros_like(self.psi)], axis=-1)
        return qid

    def axis(self, qid: int) -> int:
        return self.ids.index(qid)

    def prob1(self, qid: int) -> float:
        ax = self.axis(qid)
        p = np.abs(np.moveaxis(self.psi, ax, 0)[1]) ** 2
        return float(p.sum())

    def observe(self, qid: int) -> int:
        """Project a deterministic qubit; raise if it is not one."""
        p1 = self.prob1(qid)
        if p1 < 1e-9:
            outcome = 0
        elif p1 > 1 - 1e-9:
            outcome = 1
        else:
            raise Nondeterministic(f"p1={p1}")
        ax = self.axis(qid)
        moved = np.moveaxis(self.psi, ax, 0)
        moved[1 - outcome] = 0
        norm = np.linalg.norm(self.psi)
        self.psi = self.psi / norm
        return outcome

    def drop(self, qid: int) -> None:
        """Remove a qubit known to be in a basis state."""
        outcome = self.observe(qid)
        ax = self.axis(qid)
        self.psi = np.moveaxis(self.psi, ax, 0)[outcome].copy()
        del self.ids[ax]

    def apply(self, unitary: np.ndarray, qids: list[int], controls: list[int]):
        k = len(qids)
        u = unitary.reshape((2,) * (2 * k))
        axes = [self.axis(q) for q in qids]
        caxes = [self.axis(c) for c in controls]
        if len(set(axes + caxes)) != k + len(caxes):
            raise Undefined("a gate names one qubit twice")
        psi = self.psi
        if controls:
            # apply only to the subspace where all controls are 1
            sub = psi
            idx = [slice(None)] * psi.ndim
            for c in caxes:
                idx[c] = 1
            sub = psi[tuple(idx)]
            # axes shift after fixing control indices
            remaining = [a for a in range(psi.ndim) if a not in caxes]
            sub_axes = [remaining.index(a) for a in axes]
            new_sub = np.tensordot(u, sub, axes=(list(range(k, 2 * k)), sub_axes))
            new_sub = np.moveaxis(new_sub, list(range(k)), sub_axes)
            psi = psi.copy()
            psi[tuple(idx)] = new_sub
            self.psi = psi
        else:
            new = np.tensordot(u, psi, axes=(list(range(k, 2 * k)), axes))
            self.psi = np.moveaxis(new, list(range(k)), axes)


def _rot(name: str, theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]])
    if name == "ry":
        return np.array([[c, -s], [s, c]])
    if name == "rz":
        return np.array(
            [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]]
        )
    raise KeyError(name)


def gate_matrix(name: str, params: list[float]) -> np.ndarray:
    n = name
    if n == "i":
        return np.eye(2)
    if n == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if n == "y":
        return np.array([[0, -1j], [1j, 0]])
    if n == "z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if n == "h":
        return np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    if n == "s":
        return np.array([[1, 0], [0, 1j]])
    if n == "t":
        return np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]])
    if n == "sqrt_x":
        return np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
    if n == "sqrt_y":
        return np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2
    if n in ("rx", "ry", "rz"):
        return _rot(n, params[0])
    if n == "r1":
        return np.array([[1, 0], [0, cmath.exp(1j * params[0])]])
    if n == "u":
        th, ph, la = params
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array(
            [
                [c, -cmath.exp(1j * la) * s],
                [cmath.exp(1j * ph) * s, cmath.exp(1j * (la + ph)) * c],
            ]
        )
    if n == "swap":
        m = np.eye(4, dtype=complex)
        m[[1, 2]] = m[[2, 1]]
        return m
    if n == "gphase":
        return np.array([[cmath.exp(1j * params[0])]])
    raise Undefined(f"unknown gate {name}")


# -- interpreter ------------------------------------------------------------


class Interp:
    def __init__(self, max_steps: int = 200_000):
        self.q = QState()
        self.steps = 0
        self.max_steps = max_steps

    # entry ---------------------------------------------------------------
    def run(self, mt: ir.Method, args: tuple):
        return self.call(mt, list(args))

    def call(self, mt: ir.Method, args: list):
        block = mt.callable_region.blocks[0]
        env: dict[ir.SSAValue, object] = {}
        params = list(block.args)[1:]
        if len(params) != len(args):
            raise Undefined("call arity")
        for p, a in zip(params, args):
            env[p] = a
        result = self.block(block, env)
        assert result[0] == "return"
        return result[1]

    def block(self, block: ir.Block, env: dict):
        for stmt in block.stmts:
            self.steps += 1
            if self.steps > self.max_steps:
                raise Undefined("step budget")
            if isinstance(stmt, stmts.Return):
                return ("return", [env[v] for v in stmt.values])
            if isinstance(stmt, stmts.Yield):
                return ("yield", [env[v] for v in stmt.values])
            outs = self.stmt(stmt, env)
            if outs is None:
                outs = ()
            if len(outs) != len(stmt.results):
                raise AssertionError(f"{stmt.name}: {len(outs)} vs {len(stmt.results)}")
            for r, o in zip(stmt.results, outs):
                env[r] = o
        raise AssertionError("block without terminator")

    def region_block(self, block: ir.Block, args: list):
        env = {}
        if len(block.args) != len(args):
            raise Undefined("region arity")
        for p, a in zip(block.args, args):
            env[p] = a
        kind, vals = self.block(block, env)
        assert kind == "yield"
        return vals

    # helpers -------------------------------------------------------------
    def fl(self, x) -> float:
        x = float(x)
        if not math.isfinite(x):
            raise Nonfinite()
        return x

    def stmt(self, s: ir.Statement, env: dict):
        def g(v):
            return env[v]

        match s:
            # constants
            case stmts.ConstInt():
                if s.bitwidth == 1:
                    return (bool(s.value),)
                return (wrap(s.value),)
            case stmts.ConstFloat():
                return (float(s.value),)
            # int binary
            case stmts.IntBinary():
                a, b = g(s.lhs), g(s.rhs)
                bit = is_bit(s.lhs)
                if bit:
                    a, b = bool(a), bool(b)
                    match s:
                        case stmts.IntAnd():
                            return (a and b,)
                        case stmts.IntOr():
                            return (a or b,)
                        case stmts.IntXor():
                            return (a != b,)
                        case stmts.IntAdd():
                            return (a != b,)
                        case stmts.IntSub():
                            return (a != b,)
                        case stmts.IntMul():
                            return (a and b,)
                        case _:
                            raise Undefined(f"{s.name} on bits")
                a, b = int(a), int(b)
                return (self.int_binary(s, a, b),)
            case stmts.IntCompare():
                a, b = g(s.lhs), g(s.rhs)
                if is_bit(s.lhs):
                    a, b = int(bool(a)), int(bool(b))
                match s:
                    case stmts.IntEq():
                        return (a == b,)
                    case stmts.IntLtS():
                        return (a < b,)
                    case stmts.IntLteS():
                        return (a <= b,)
                    case stmts.IntLtU():
                        return (pattern(a) < pattern(b),)
                    case stmts.IntLteU():
                        return (pattern(a) <= pattern(b),)
            case stmts.IntNot():
                v = g(s.value)
                if is_bit(s.value):
                    return (not bool(v),)
                return (wrap(~int(v)),)
            case stmts.IntAbs():
                v = g(s.value)
                if is_bit(s.value):
                    return (bool(v),)
                return (wrap(abs(int(v))),)
            case stmts.IntSelect():
                return (g(s.yes) if bool(g(s.condition)) else g(s.no),)
            case stmts.FloatSelect():
                return (g(s.yes) if bool(g(s.condition)) else g(s.no),)
            # float
            case stmts.FloatBinary():
                a, b = float(g(s.lhs)), float(g(s.rhs))
                return (self.float_binary(s, a, b),)
            case stmts.FloatCompare():
                a, b = float(g(s.lhs)), float(g(s.rhs))
                match s:
                    case stmts.FloatEq():
                        return (a == b,)
                    case stmts.FloatLt():
                        return (a < b,)
                    case stmts.FloatLte():
                        return (a <= b,)
            case stmts.FloatPredicate():
                a = float(g(s.value))
                if isinstance(s, stmts.FloatIsNan):
                    return (math.isnan(a),)
                return (math.isinf(a),)
            case stmts.FloatUnary():
                a = float(g(s.value))
                return (self.float_unary(s, a),)
            # int arrays
            case stmts.IntArrayConst():
                if s.bitwidth == 1:
                    return ([bool(v) for v in s.values],)
                return ([wrap(v) for v in s.values],)
            case stmts.IntArrayZero():
                n = int(g(s.size))
                if n < 0:
                    raise Undefined("negative array size")
                return ([False] * n if s.bitwidth == 1 else [0] * n,)
            case stmts.IntArrayGet():
                arr, i = g(s.array), int(g(s.index))
                if not 0 <= i < len(arr):
                    raise Undefined("array index")
                return (arr[i],)
            case stmts.IntArraySet():
                arr, i, v = list(g(s.array)), int(g(s.index)), g(s.value)
                if not 0 <= i < len(arr):
                    raise Undefined("array index")
                arr[i] = v
                return (arr,)
            case stmts.IntArrayLen():
                return (len(g(s.array)),)
            case stmts.IntArrayCreate():
                return ([g(v) for v in s.values],)
            case stmts.FloatArrayConst():
                return ([float(v) for v in s.values],)
            case stmts.FloatArrayZero():
                n = int(g(s.size))
                if n < 0:
                    raise Undefined("negative array size")
                return ([0.0] * n,)
            case stmts.FloatArrayGet():
                arr, i = g(s.array), int(g(s.index))
                if not 0 <= i < len(arr):
                    raise Undefined("array index")
                return (arr[i],)
            case stmts.FloatArraySet():
                arr, i, v = list(g(s.array)), int(g(s.index)), g(s.value)
                if not 0 <= i < len(arr):
                    raise Undefined("array index")
                arr[i] = v
                return (arr,)
            case stmts.FloatArrayLen():
                return (len(g(s.array)),)
            case stmts.FloatArrayCreate():
                return ([g(v) for v in s.values],)
            # control flow
            case stmts.For():
                start, stop, step = int(g(s.start)), int(g(s.stop)), int(g(s.step))
                if step == 0:
                    raise Undefined("for step 0")
                state = [g(v) for v in s.state]
                body = s.body.blocks[0]
                for i in range(start, stop, step):
                    state = self.region_block(body, [i, *state])
                return tuple(state)
            case stmts.Switch():
                sel = int(g(s.selector))
                inputs = [g(v) for v in s.inputs]
                if 0 <= sel < len(s.branches):
                    block = s.branches[sel].blocks[0]
                else:
                    block = s.default.blocks[0]
                return tuple(self.region_block(block, inputs))
            case stmts.While():
                state = [g(v) for v in s.inputs]
                while True:
                    out = self.region_block(s.before.blocks[0], state)
                    cond, outs = out[0], out[1:]
                    if not bool(cond):
                        return tuple(outs)
                    state = self.region_block(s.after.blocks[0], outs)
            case stmts.Call():
                return tuple(self.call(s.callee, [g(v) for v in s.inputs]))
            # qubits
            case stmts.Alloc():
                return (self.q.alloc(),)
            case stmts.Free():
                self.q.drop(g(s.wire))
                return ()
            case stmts.FreeZero():
                qid = g(s.wire)
                if self.q.observe(qid) != 0:
                    raise Undefined("free_zero of |1>")
                self.q.drop(qid)
                return ()
            case stmts.Reset():
                qid = g(s.wire)
                if self.q.observe(qid) == 1:
                    self.q.apply(gate_matrix("x", []), [qid], [])
                return (qid,)
            case stmts.MeasureNd():
                qid = g(s.wire)
                return (qid, bool(self.q.observe(qid)))
            case stmts.Measure():
                qid = g(s.wire)
                bit = bool(self.q.observe(qid))
                self.q.drop(qid)
                return (bit,)
            case stmts.Gate():
                return self.gate(s, env)
            case stmts.Ppr():
                raise Undefined("ppr not modelled")
            # registers: a register is a list of qubit id or None (empty slot)
            case stmts.RegAlloc():
                n = int(g(s.size))
                if n < 0:
                    raise Undefined("negative register size")
                return ([self.q.alloc() for _ in range(n)],)
            case stmts.RegCreate():
                return ([g(w) for w in s.wires],)
            case stmts.RegFree() | stmts.RegFreeZero():
                for qid in g(s.reg):
                    if qid is not None:
                        if (
                            isinstance(s, stmts.RegFreeZero)
                            and self.q.observe(qid) != 0
                        ):
                            raise Undefined("reg_free_zero of |1>")
                        self.q.drop(qid)
                return ()
            case stmts.Extract():
                reg, i = list(g(s.reg)), int(g(s.index))
                if not 0 <= i < len(reg) or reg[i] is None:
                    raise Undefined("extract from empty/out-of-range slot")
                qid = reg[i]
                reg[i] = None
                return (reg, qid)
            case stmts.Insert():
                reg, i, qid = list(g(s.reg)), int(g(s.index)), g(s.wire)
                if not 0 <= i < len(reg) or reg[i] is not None:
                    raise Undefined("insert into filled/out-of-range slot")
                reg[i] = qid
                return (reg,)
            case stmts.ExtractSlice():
                reg, a, n = list(g(s.reg)), int(g(s.start)), int(g(s.length))
                if n < 0 or a < 0 or a + n > len(reg):
                    raise Undefined("slice range")
                part = reg[a : a + n]
                if any(q is None for q in part):
                    raise Undefined("slice over an empty slot")
                for k in range(a, a + n):
                    reg[k] = None
                return (reg, part)
            case stmts.InsertSlice():
                reg, a, part = list(g(s.reg)), int(g(s.start)), list(g(s.slice_reg))
                n = len(part)
                if a < 0 or a + n > len(reg):
                    raise Undefined("slice range")
                if any(reg[k] is not None for k in range(a, a + n)):
                    raise Undefined("insert slice over a filled slot")
                reg[a : a + n] = part
                return (reg,)
            case stmts.RegSplit():
                reg, i = list(g(s.reg)), int(g(s.index))
                if not 0 <= i <= len(reg):
                    raise Undefined("split index")
                return (reg[:i], reg[i:])
            case stmts.RegJoin():
                return (list(g(s.first)) + list(g(s.second)),)
            case stmts.RegLength():
                reg = g(s.reg)
                return (reg, len(reg))
        raise Undefined(f"unhandled statement {s.name}")

    def int_binary(self, s, a: int, b: int) -> int:
        match s:
            case stmts.IntAdd():
                return wrap(a + b)
            case stmts.IntSub():
                return wrap(a - b)
            case stmts.IntMul():
                return wrap(a * b)
            case stmts.IntDivS():
                if b == 0:
                    raise Undefined("div by zero")
                q = abs(a) // abs(b)
                return wrap(q if (a < 0) == (b < 0) else -q)
            case stmts.IntRemS():
                if b == 0:
                    raise Undefined("rem by zero")
                q = abs(a) // abs(b)
                q = q if (a < 0) == (b < 0) else -q
                return wrap(a - b * q)
            case stmts.IntDivU():
                if b == 0:
                    raise Undefined("div by zero")
                return wrap(pattern(a) // pattern(b))
            case stmts.IntRemU():
                if b == 0:
                    raise Undefined("rem by zero")
                return wrap(pattern(a) % pattern(b))
            case stmts.IntPow():
                if b < 0:
                    raise Undefined("negative exponent")
                return wrap(a**b)
            case stmts.IntAnd():
                return wrap(a & b)
            case stmts.IntOr():
                return wrap(a | b)
            case stmts.IntXor():
                return wrap(a ^ b)
            case stmts.IntMinS():
                return min(a, b)
            case stmts.IntMaxS():
                return max(a, b)
            case stmts.IntMinU():
                return a if pattern(a) <= pattern(b) else b
            case stmts.IntMaxU():
                return a if pattern(a) >= pattern(b) else b
            case stmts.IntShl():
                if not 0 <= b < BITS:
                    raise Undefined("shift amount")
                return wrap(pattern(a) << b)
            case stmts.IntShr():
                if not 0 <= b < BITS:
                    raise Undefined("shift amount")
                return wrap(pattern(a) >> b)
        raise Undefined(s.name)

    def float_binary(self, s, a: float, b: float) -> float:
        try:
            match s:
                case stmts.FloatAdd():
                    r = a + b
                case stmts.FloatSub():
                    r = a - b
                case stmts.FloatMul():
                    r = a * b
                case stmts.FloatPow():
                    r = math.pow(a, b)
                case stmts.FloatAtan2():
                    r = math.atan2(a, b)
                case stmts.FloatMax():
                    r = max(a, b)
                case stmts.FloatMin():
                    r = min(a, b)
                case _:
                    raise Undefined(s.name)
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            raise Nonfinite(str(e))
        return self.fl(r)

    def float_unary(self, s, a: float) -> float:
        fn = {
            stmts.FloatSqrt: math.sqrt,
            stmts.FloatAbs: math.fabs,
            stmts.FloatCeil: lambda x: float(math.ceil(x)),
            stmts.FloatFloor: lambda x: float(math.floor(x)),
            stmts.FloatExp: math.exp,
            stmts.FloatLog: math.log,
            stmts.FloatSin: math.sin,
            stmts.FloatCos: math.cos,
            stmts.FloatTan: math.tan,
            stmts.FloatAsin: math.asin,
            stmts.FloatAcos: math.acos,
            stmts.FloatAtan: math.atan,
            stmts.FloatSinh: math.sinh,
            stmts.FloatCosh: math.cosh,
            stmts.FloatTanh: math.tanh,
            stmts.FloatAsinh: math.asinh,
            stmts.FloatAcosh: math.acosh,
            stmts.FloatAtanh: math.atanh,
        }[type(s)]
        try:
            r = fn(a)
        except (ValueError, OverflowError) as e:
            raise Nonfinite(str(e))
        return self.fl(r)

    def gate(self, s: stmts.Gate, env: dict):
        shape = canonical(s)
        params = [float(env[p]) for p in s.params]
        targets = [env[w] for w in shape.targets]
        controls = [env[w] for w in shape.controls]
        if shape.power < 0:
            raise Undefined("negative power")
        m = gate_matrix(shape.name, params)
        if shape.adjoint:
            m = m.conj().T
        if shape.name == "gphase":
            if targets or controls:
                raise Undefined("gphase with qubits")
            # a global phase is unobservable; controlled gphase is a phase on controls
            return ()
        if m.shape[0] != 2 ** len(targets):
            raise Undefined(f"gate {shape.name} arity")
        for _ in range(shape.power):
            self.q.apply(m, targets, controls)
        # results: output wires of targets, then controls, in the *statement's*
        # argument order (wire identity is preserved)
        return tuple(env[w] for w in (*s.targets, *s.controls))
