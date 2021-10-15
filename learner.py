import itertools as it
from sympy import symbols, lambdify, sympify, Symbol
import numpy as np
import logging

from cpmpy import *

from cpmpy.transformations.flatten_model import get_or_make_var

logger = logging.getLogger(__name__)


def pairs(example):
    for [x, y] in it.product(example, repeat=2):
        yield x, y


def index_pairs(exampleLen):
    for (i, j) in it.combinations(range(exampleLen), r=2):
        yield i, j


def unary_operators():
    def modulo(x):
        return x % 2

    def power(x):
        return x * x

    def identity(x):
        return x

    for f in [identity, abs]:
        yield f


def generate_unary_exp(x):
    for u in unary_operators():
        yield u(x)


def binary_operators(x, y):
    for u in unary_operators():
        yield u(x) + u(y)
        yield u(x) - u(y)
        yield u(y) - u(x)


def generate_binary_expr(x, y):
    for b in binary_operators(x, y):
        for u in generate_unary_exp(b):  # redundancy due to identity
            yield u


def constraint_learner(solutions, n_vars):
    bounds = dict()
    x, y = symbols("x y")
    for u in generate_unary_exp(x):
        k = str(u)
        bounds[k] = dict()
        f = lambdify(x, u, "math")
        for i in range(n_vars):
            bounds[k][(i,)] = {}
            vals = f(solutions[:, i])
            bounds[k][(i,)]["l"] = min(vals)
            bounds[k][(i,)]["u"] = max(vals)

    for b in generate_binary_expr(x, y):
        k = str(b)
        bounds[k] = dict()
        f = lambdify([x, y], b, "math")
        for (i, j) in index_pairs(n_vars):
            bounds[k][(i, j)] = {}
            vals = f(solutions[:, i], solutions[:, j])
            bounds[k][(i, j)]["l"] = min(vals)
            bounds[k][(i, j)]["u"] = max(vals)
    return bounds


def filter_negatives(negData, lb, ub):  # InComplete
    x, y = symbols("x y")
    for u in generate_unary_exp(x):
        k = str(u)

        for i in range(len(lb[k])):
            breaksLB = 0
            for example in negData:
                if u.subs({x: example[i]}) < lb[k][i]:
                    breaksLB = 1
                    break
            if breaksLB == 0:
                del lb[k]

            breaksUB = 0
            for example in negData:
                if u.subs({x: example[i]}) > ub[k][i]:
                    breaksUB = 1
                    break
            if breaksUB == 0:
                del ub[k]

    # for b in generate_binary_expr(x,y):
    #     k=str(b)
    #
    #     for i in range(len(lb[k])):
    #         breaksLB = 0
    #         for example in negData:
    #             if b.subs({x:v1, y: v2})<lb[k][i]:
    #                 breaksLB=1
    #                 break
    #         if breaksLB==1:
    #             break
    #     if breaksLB==0:
    #         del lb[k]
    #
    #     breaksUB = 0
    #     for example in negData:
    #         for i,(v1, v2) in enumerate(pairs(example)):
    #             if b.subs({x: v1, y: v2})>ub[k][i]:
    #                 breaksUB = 1
    #                 break
    #         if breaksUB == 1:
    #             break
    #     if breaksUB == 0:
    #         del ub[k]
    #
    # for u in generate_unary_exp(x):
    #     for b in generate_binary_expr(x,y):
    #         k = str(u)
    #         k = k.replace('x', '(' + str(b) + ')')
    #         breaksLB = 0
    #         for example in negData:
    #             for i, (v1, v2) in enumerate(pairs(example)):
    #                 if u.subs({x: b.subs({x: v1, y: v2})}) < lb[k][i]:
    #                     breaksLB = 1
    #                     break
    #             if breaksLB == 1:
    #                 break
    #         if breaksLB == 0:
    #             del lb[k]
    #
    #         breaksUB = 0
    #         for example in negData:
    #             for i, (v1, v2) in enumerate(pairs(example)):
    #                 if u.subs({x: b.subs({x: v1, y: v2})}) > ub[k][i]:
    #                     breaksUB = 1
    #                     break
    #             if breaksUB == 1:
    #                 break
    #         if breaksUB == 0:
    #             del ub[k]
    return lb, ub


def create_variables(var_bounds, name):
    return [
        intvar(lb, ub, name=f"{name}[{i}]")
        for i, (lb, ub) in enumerate(var_bounds)
    ]


def create_model(var_bounds, expr_bounds, name):
    x, y = symbols("x y")
    cp_vars = create_variables(var_bounds, name)

    m = Model()
    for expr, inst in expr_bounds.items():
        for (index), values in inst.items():
            lb = values["l"]
            ub = values["u"]
            if len(index) == 1:
                e = sympify(expr)
                f = lambdify(x, e)
                cpm_e = f(cp_vars[index[0]])
                (v, _) = get_or_make_var(cpm_e)
                if lb != v.lb:
                    m += [cpm_e >= lb]
                if ub != v.ub:
                    m += [cpm_e <= ub]
            else:
                e = sympify(expr)
                f = lambdify([x, y], e)
                cpm_e = f(cp_vars[index[0]], cp_vars[index[1]])
                (v, _) = get_or_make_var(cpm_e)
                if lb != v.lb:
                    m += [cpm_e >= lb]
                if ub != v.ub:
                    m += [cpm_e <= ub]

    return m, cp_vars


def is_sat(m, m_vars, sols, exp, objectives=None):
    sats = []
    for i, sol in enumerate(sols):
        m2 = Model([c for c in m.constraints])
        m2 += m_vars == sol
        sat = m2.solve()
        if objectives is not None and sat:
            sat = exp(sol) == objectives[i]
        sats.append(sat)
    return sats


def check_solutions(m, mvars, sols, exp, objectives=None):
    if len(sols) == 0:
        print("No solutions to check")
        return 1.0

    sats = is_sat(m, mvars, sols, exp, objectives)
    logger.info(f"{sum(sats)} satisfied out of {len(sats)}")
    return sum(sats) * 100.0 / len(sats)


def check_obective(exp, sols, objectives, verbose=False):
    if len(sols) == 0:
        print("No solutions to check")
        return 1.0
    sats = []
    for i, sol in enumerate(sols):
        sat = exp(sol) == objectives[i]
        sats.append(sat)

        if verbose:
            if sat:
                print(f"Sol {sol} indeed satisfies the objective {exp(sol)}")
            else:
                print(f"!!! Sol {sol} does not satisfy the objective {exp(sol)}")
    print(f"{sum(sats)} objectives satisfied out of {len(sats)}")
    return sum(sats) * 100.0 / len(sats)


def strip_empty_entries(dictionary):
    new_data = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v = strip_empty_entries(v)
        if v not in ("", None, {}):
            new_data[k] = v
    return new_data


def filter_redundant(var_bounds, expr_bounds, name, inputData, propositional=False):
    mapping = None
    if propositional:
        m, _ = create_model(var_bounds, expr_bounds, name=name)
        # reverse, so more complex are eliminated first
        constraints = reversed(m.constraints)
    else:
        m, _, mapping = create_gen_model(var_bounds, expr_bounds, name=name, inputData=inputData)
        constraints = m.constraints

    constraints = [c for c in constraints]  # take copy

    i = 0
    while i < len(constraints):
        m2 = Model(constraints[:i] + constraints[i + 1 :])
        m2 += ~all(constraints[i])
        if m2.solve():
            i += 1
        else:
            del constraints[i]

            if mapping:
                del expr_bounds[mapping[i][0]][mapping[i][1]][mapping[i][2]]
                del mapping[i]

    expr_bounds = strip_empty_entries(expr_bounds)
    return expr_bounds, constraints



def generate_unary_sequences(n):
    def even(n):
        lst = []
        for i in range(0, n, 2):
            lst.append((i,))
        return lst

    def odd(n):
        lst = []
        for i in range(1, n, 2):
            lst.append((i,))
        return lst

    def series(n):
        lst = []
        for i in range(0, n):
            lst.append((i,))
        return lst

    lst = {}

    if even(n):
        lst["evenUn"] = even(n)
    if odd(n):
        lst["oddUn"] = odd(n)
    if series(n):
        lst["seriesUn"] = series(n)
    return lst


def generate_binary_sequences(n, data=None):
    def even(n):
        lst = []
        for i in range(0, n - 2, 2):
            lst.append((i, i + 2))
        return lst

    def odd(n):
        lst = []
        for i in range(1, n - 2, 2):
            lst.append((i, i + 2))
        return lst

    def series(n):
        lst = []
        for i in range(0, n - 1):
            lst.append((i, i + 1))
        return lst

    def all_pairs(n):
        return list(it.combinations(range(n), r=2))


    lst = {}
    if even(n):
        lst["evenBin"] = even(n)
    if odd(n):
        lst["oddBin"] = odd(n)
    if series(n):
        lst["seriesBin"] = series(n)
    if all_pairs(n):
        lst["allBin"] = all_pairs(n)
    if data:
        lst["jsonSeq"] = data
    return lst


def generalise_bounds(bounds, size, inputData):
    generalBounds = {}
    unSeq = generate_unary_sequences(size)
    binSeq = generate_binary_sequences(size, inputData)
    x, y = symbols("x y")
    for b in generate_binary_expr(x, y):
        exp = str(b)
        generalBounds[exp] = {}
        for k, seq in binSeq.items():
            generalBounds[exp][k] = {}
            tmp = np.array(
                [[bounds[exp][tple]["l"], bounds[exp][tple]["u"]] for tple in seq]
            )
            generalBounds[exp][k]["l"] = min(tmp[:, 0])
            generalBounds[exp][k]["u"] = max(tmp[:, 1])

    for u in generate_unary_exp(x):
        exp = str(u)
        generalBounds[exp] = {}
        for k, seq in unSeq.items():
            generalBounds[exp][k] = {}
            tmp = np.array(
                [[bounds[exp][tple]["l"], bounds[exp][tple]["u"]] for tple in seq]
            )
            generalBounds[exp][k]["l"] = min(tmp[:, 0])
            generalBounds[exp][k]["u"] = max(tmp[:, 1])
    return generalBounds


def create_gen_model(var_bounds, genBounds, name, inputData):
    cp_vars = create_variables(var_bounds, name)
    size = len(cp_vars)
    unSeq = generate_unary_sequences(size)
    binSeq = generate_binary_sequences(size, inputData)
    x, y = symbols("x y")
    m = Model()
    mapping = []
    for expr, inst in genBounds.items():
        e = sympify(expr)
        numSym = len(e.atoms(Symbol))
        if numSym == 1:
            for seq, values in inst.items():
                constraints_l = []
                constraints_u = []
                for index in unSeq[seq]:
                    f = lambdify(x, e)
                    cpm_e = f(cp_vars[index[0]])
                    (v, _) = get_or_make_var(cpm_e)
                    if "l" in values:
                        # m += [cpm_e >= values['l']]
                        constraints_l.append(cpm_e >= values["l"])
                    if "u" in values:
                        # m += [cpm_e <= values['u']]
                        constraints_u.append(cpm_e <= values["u"])
                # print(tmp)
                if constraints_l:
                    mapping.append([expr, seq, "l"])
                    m += constraints_l
                if constraints_u:
                    m += constraints_u
                    mapping.append([expr, seq, "u"])
        else:
            for seq, values in inst.items():
                constraints_l = []
                constraints_u = []
                for index in binSeq[seq]:
                    f = lambdify([x, y], e)
                    cpm_e = f(cp_vars[index[0]], cp_vars[index[1]])
                    (v, _) = get_or_make_var(cpm_e)
                    if "l" in values:
                        # m += [cpm_e >= values['l']]
                        constraints_l.append(cpm_e >= values["l"])
                    if "u" in values:
                        # m += [cpm_e <= values['u']]
                        constraints_u.append(cpm_e <= values["u"])
                if constraints_l:
                    m += constraints_l
                    mapping.append([expr, seq, "l"])
                if constraints_u:
                    m += constraints_u
                    mapping.append([expr, seq, "u"])
    return m, cp_vars, mapping


def filter_trivial(var_bounds, genBounds, size, name, inputData):
    cp_vars = create_variables(var_bounds, name)

    unSeq = generate_unary_sequences(size)
    binSeq = generate_binary_sequences(size, inputData)
    x, y = symbols("x y")
    for expr, inst in genBounds.items():
        e = sympify(expr)
        numSym = len(e.atoms(Symbol))
        if numSym == 1:
            for seq, values in inst.items():
                lb = values["l"]
                ub = values["u"]
                for index in unSeq[seq]:
                    f = lambdify(x, e)
                    cpm_e = f(cp_vars[index[0]])
                    (v, _) = get_or_make_var(cpm_e)
                    if lb == v.lb:
                        del genBounds[expr][seq]["l"]
                        break
                    if ub == v.ub:
                        del genBounds[expr][seq]["u"]
                        break
        else:
            for seq, values in inst.items():
                lb = values["l"]
                ub = values["u"]
                for index in binSeq[seq]:
                    f = lambdify([x, y], e)
                    cpm_e = f(cp_vars[index[0]], cp_vars[index[1]])
                    (v, _) = get_or_make_var(cpm_e)
                    if lb == v.lb:
                        del genBounds[expr][seq]["l"]
                        break
                    if ub == v.ub:
                        del genBounds[expr][seq]["u"]
                        break
    return genBounds
