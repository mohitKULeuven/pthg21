
import itertools as it
from sympy import symbols, lambdify, sympify
import json
import numpy as np
from cpmpy import *
from musx import musx
from cpmpy.solvers import CPM_ortools

def pairs(example):
    for [x, y] in it.product(example, repeat=2):
        yield x,y

def index_pairs(exampleLen):
    for (i, j) in it.combinations(range(exampleLen), r=2):
        yield i, j


def unary_operators():
    def modulo(x):
        return x % 2

    # def power(x):
    #     return x*x

    def identity(x):
        return x

    for f in [identity, abs, modulo]:
        yield f

def generate_unary_exp(x):
    for u in unary_operators():
        yield u(x)

def binary_operators(x,y):
    for u in unary_operators():
        yield u(x) + u(y)
        yield u(x) - u(y)
        yield u(y) - u(x)

def generate_binary_expr(x, y):
    for b in binary_operators(x,y):
        yield b

    for b in binary_operators(x,y):
        for u in generate_unary_exp(b): #redundancy due to identity
            yield u

def constraint_learner(solutions, n_vars):
    bounds = dict()
    x, y = symbols('x y')
    for u in generate_unary_exp(x):
        k = str(u)
        bounds[k]=dict()
        f = lambdify(x, u, "math")
        for i in range(n_vars):
            vals = f(solutions[:, i])
            bounds[k][(i,)] = (min(vals), max(vals))

    for b in generate_binary_expr(x, y):
        k = str(b)
        bounds[k] = dict()
        f = lambdify([x,y], b, "math")
        for (i,j) in index_pairs(n_vars):
            vals = f(solutions[:, i], solutions[:, j])
            bounds[k][(i,j)] = (min(vals), max(vals))
    return bounds

def filter_negatives(negData, lb, ub):
    x, y = symbols('x y')
    for u in generate_unary_exp(x):
        k=str(u)

        for i in range(len(lb[k])):
            breaksLB = 0
            for example in negData:
                if u.subs({x: example[i]})<lb[k][i]:
                    breaksLB=1
                    break
            if breaksLB==0:
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
    x, y = symbols('x y')
    cpvars = []
    for vdict in data['formatTemplate']['list']:
        # {'high': 10, 'low': 1, 'type': 'dvar'}
        cpvars.append(intvar(vdict['low'], vdict['high']))
    cpvars = cpm_array(cpvars)  # make it a CPM/Numpy array

    m = Model()
    for expr, inst in bounds.items():
        # print(expr)
        for (index), (lb, ub) in inst.items():
            if len(index)==1:
                e = sympify(expr)
                f = lambdify(x, e)
                cpm_e = f(cpvars[index[0]])
                m += [cpm_e >= lb, cpm_e <= ub]
            else:
                e = sympify(expr)
                f = lambdify([x,y], e)
                cpm_e = f(cpvars[index[0]], cpvars[index[1]])
                m += [cpm_e >= lb, cpm_e <= ub]

    unsat_cons = musx(m.constraints)
    model2 = Model(unsat_cons)
    print(model2)
    # model2.solve()
    # print(model2.status())
    # cpvars.value()


if __name__ == '__main__':
    # constraint_learner()
    # x, y = symbols('x y')
    # for b in generate_binary_expr(x,y):
    #     print(b,":", b.subs({x:5, y: 10}))

    data = json.load(open('instances/type01/instance0.json'))
    posData = np.array([d['list'] for d in data['solutions']])
    negData = np.array([d['list'] for d in data['nonSolutions']])
    print("number of solutions: ", len(posData))
    print("number of non-solutions: ", len(negData))
    bounds = constraint_learner(posData, posData.shape[1])
    print(len(bounds))
    create_model(data, bounds)
    # print(len(bounds))
    # lb,ub = filter_negatives(negData, lb, ub)
    # print(len(lb), len(ub))
