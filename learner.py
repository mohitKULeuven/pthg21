import sys
import itertools as it
from sympy import symbols, lambdify, sympify, Symbol
import json
import numpy as np
from cpmpy import *
import glob
import csv
import minizinc
from musx import musx
from cpmpy.solvers import CPM_ortools

from cpmpy.transformations.flatten_model import get_or_make_var

# from cpmpy.expressions.variables import _NumVarImpl
# def pow(self, y=2):
#     assert (y==2)
#     return self*self
# _NumVarImpl.__pow__ = pow


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
        yield b

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
            vals = f(solutions[:, i])
            bounds[k][(i,)] = (min(vals), max(vals))

    for b in generate_binary_expr(x, y):
        k = str(b)
        bounds[k] = dict()
        f = lambdify([x, y], b, "math")
        for (i, j) in index_pairs(n_vars):
            vals = f(solutions[:, i], solutions[:, j])
            bounds[k][(i, j)] = (min(vals), max(vals))
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


def create_model(data, bounds):
    x, y = symbols("x y")
    cpvars = []
    for i,vdict in enumerate(data["formatTemplate"]["list"]):
        # {'high': 10, 'low': 1, 'type': 'dvar'}
        cpvars.append(intvar(vdict["low"], vdict["high"], name=f"list[{i}]"))
    cpvars = cpm_array(cpvars)  # make it a CPM/Numpy array

    m = Model()
    for expr, inst in bounds.items():
        for (index), (lb, ub) in inst.items():
            if len(index) == 1:
                e = sympify(expr)
                f = lambdify(x, e)
                cpm_e = f(cpvars[index[0]])
                (v, _) = get_or_make_var(cpm_e)
                if lb != v.lb:
                    m += [cpm_e >= lb]
                if ub != v.ub:
                    m += [cpm_e <= ub]
            else:
                e = sympify(expr)
                f = lambdify([x, y], e)
                cpm_e = f(cpvars[index[0]], cpvars[index[1]])
                (v, _) = get_or_make_var(cpm_e)
                if lb != v.lb:
                    m += [cpm_e >= lb]
                if ub != v.ub:
                    m += [cpm_e <= ub]

    return m, cpvars
    # unsat_cons = musx(m.constraints)
    # model2 = Model(unsat_cons)
    # print(model2)
    # m.solve()
    # print(m.status())
    # print(cpvars.value())


def check_solutions(m, mvars, sols, exp, objectives=None, verbose=False):
    if len(sols) == 0:
        print("No solutions to check")
        return 1.0

    sats = []
    for i, sol in enumerate(sols):
        m2 = Model([c for c in m.constraints])
        m2 += mvars == sol
        sat = m2.solve()
        if objectives is not None and sat:
            sat = exp(sol) == objectives[i]
        sats.append(sat)

        if verbose:
            if sat:
                print(f"Sol {sol} indeed satisfies")
            else:
                print(f"!!! Sol {sol} does not satisfy after all")
    print(f"{sum(sats)} satisfied out of {len(sats)}")
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


def filter_redundant(m):
    relcons = [c for c in m.constraints]  # take copy
    relcons = relcons[::-1]  # reverse, so more complex are eliminated first
    i = 0
    while i < len(relcons):  # relcons will shrink
        # print("Checking redundancy of", relcons[i])
        m2 = Model(relcons[:i] + relcons[i + 1 :])
        m2 += ~all(relcons[i])
        if m2.solve():
            i += 1
        else:
            del relcons[i]
            # keep i, will point to next
    return Model(relcons)


def generate_json_sequence(data):
    lst = []
    inputData = data["inputData"]
    for d in inputData:
        lst.append(tuple(d.values()))
    return lst


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
    lst["evenUn"]=even(n)
    lst["oddUn"]=odd(n)
    lst["seriesUn"]=series(n)
    return lst

def generate_binary_sequences(n):
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
    lst["evenBin"]=even(n)
    lst["oddBin"]=odd(n)
    lst["seriesBin"]=series(n)
    lst["allBin"]=all_pairs(n)
    return lst

def generalise_bounds(bounds, size):
    generalBounds={}
    unSeq=generate_unary_sequences(size)
    binSeq=generate_binary_sequences(size)
    x, y = symbols("x y")
    for b in generate_binary_expr(x, y):
        exp = str(b)
        generalBounds[exp]={}
        for k, seq in binSeq.items():
            tmp=np.array([bounds[exp][tple] for tple in seq])
            generalBounds[exp][k] = (min(tmp[:,0]), max(tmp[:,1]))

    for u in generate_unary_exp(x):
        exp = str(u)
        generalBounds[exp]={}
        for k, seq in unSeq.items():
            tmp=np.array([bounds[exp][tple] for tple in seq])
            generalBounds[exp][k] = (min(tmp[:,0]), max(tmp[:,1]))
    return generalBounds


def create_gen_model(data, genBounds, size):
    x, y = symbols("x y")
    cpvars = []
    for i,vdict in enumerate(data["formatTemplate"]["list"]):
        cpvars.append(intvar(vdict["low"], vdict["high"], name=f"list[{i}]"))
    cpvars = cpm_array(cpvars)
    unSeq = generate_unary_sequences(size)
    binSeq = generate_binary_sequences(size)
    x, y = symbols("x y")
    m = Model()
    for expr, inst in genBounds.items():
        e = sympify(expr)
        numSym = len(e.atoms(Symbol))
        if numSym==1:
            for seq, (lb, ub) in inst.items():
                for (index) in unSeq[seq]:
                    f = lambdify(x, e)
                    cpm_e = f(cpvars[index[0]])
                    (v, _) = get_or_make_var(cpm_e)
                    if lb != v.lb:
                        m += [cpm_e >= lb]
                    if ub != v.ub:
                        m += [cpm_e <= ub]
        else:
            for seq, (lb, ub) in inst.items():
                for (index) in binSeq[seq]:
                    f = lambdify([x, y], e)
                    cpm_e = f(cpvars[index[0]], cpvars[index[1]])
                    (v, _) = get_or_make_var(cpm_e)
                    if lb != v.lb:
                        m += [cpm_e >= lb]
                    if ub != v.ub:
                        m += [cpm_e <= ub]
    return m, cpvars

if __name__ == "__main__":
    args = sys.argv[1:]
    # t=int(args[0])
    for t in [1,2,4,7,8,13,14,15,16]:
        csvfile = open(f"type{t:02d}.csv", "w")
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(["file", "constraints", "filtered_constraints", "num_pos", "percentage_pos",
                             "num_neg", "percentage_neg"])
        path = f"instances/type{t:02d}/inst*.json"
        files = glob.glob(path)
        for file in files:
            print(file)
            data = json.load(open(file))
            # data = json.load(open(f"instances/type0{args[0]}/instance{args[1]}.json"))
            if data["solutions"]:
                posData = np.array([np.array(d["list"]).flatten() for d in data["solutions"]])
                negData = np.array([np.array(d["list"]).flatten() for d in data["nonSolutions"]])

                bounds = constraint_learner(posData, posData.shape[1])
                # genBounds = generalise_bounds(bounds, posData.shape[1])
                # numConstr = 0
                # for k, v in bounds.items():
                #     numConstr += len(v)
                # print(f"learned {numConstr} constraints from {len(bounds)} different expressions")

                m, mvars = create_model(data, bounds)
                num_cons=len(m.constraints)
                # m, mvars = create_gen_model(data, genBounds, posData.shape[1])
                # print(f"number of constraints in the model: {len(m.constraints)}")
                posDataObj, negDataObj = None, None
                if 'objective' in data['solutions'][0]:
                    posDataObj = np.array([d["objective"] for d in data["solutions"]])
                    negDataObj = np.array([d["objective"] for d in data["nonSolutions"]])
                m = filter_redundant(m)
                print(
                    f"number of constraints in the model after redundancy check: {len(m.constraints)}"
                )
                print(m)
                perc_pos=check_solutions(m, mvars, posData, max, posDataObj)
                perc_neg=100-check_solutions(m, mvars, negData, max, negDataObj)
                filewriter.writerow([file,num_cons,len(m.constraints),len(posData),perc_pos, len(negData), perc_neg])
    csvfile.close()

    # check_obective(max, negData, negDataObj)
    # print(len(bounds))
    # lb,ub = filter_negatives(negData, lb, ub)
    # print(len(lb), len(ub))
