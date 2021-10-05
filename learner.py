
import itertools as it
from sympy import symbols

def pairs(example):
    for [x, y] in it.product(example, repeat=2):
        yield x,y


def unary_operators():
    def modulo(x):
        return x % 2

    def power(x):
        return x*x

    def identity(x):
        return x

    for f in [identity, abs, power, modulo]:
        yield f

def generate_unary_exp(x):
    for u in unary_operators():
        yield u(x)

def generate_binary_expr(x,y):
    for u in unary_operators():
        yield u(x) + u(y)
        yield u(x) - u(y)
        yield u(y) - u(x)

def generate_expressions(x, y):
    for u in generate_unary_exp(x):
        yield u

    for b in generate_binary_expr(x,y):
        yield b

    for b in generate_binary_expr(x,y):
        for u in generate_unary_exp(b):
            yield u

def constraint_learner(instance):
    lb= {}
    ub = {}
    x, y = symbols('x y')
    for u in generate_unary_exp(x):
        if str(u) not in lb:
            lb[str(u)]=[]
            ub[str(u)]=[]
        for v in instance:
            val = u.subs({x: v})
            lb[str(u)].append(val)
            ub[str(u)].append(val)

    for b in generate_binary_expr(x,y):
        if str(b) not in lb:
            lb[str(b)] = []
            ub[str(b)] = []
        for (x,y) in pairs(instance):
            val=b.subs({x:x, y: y})
            lb[str(u)].append(val)
            ub[str(u)].append(val)

    for u in generate_unary_exp(x):
        for b in generate_binary_expr(x,y):
            k = str(u)+"("+str(b)+")"
            if k not in lb:
                lb[k] = []
                ub[k] = []
            for (x, y) in pairs(instance):
                val = u(b.subs({x:x, y: y}))
                lb[k].append(val)
                ub[k].append(val)
    return lb,ub

# def filter_negatives(negData, unaryExpressions, binaryExpressions, lb, ub):
#     for i,u in enumerate(unaryExpressions):
#         k=str(i)+","
#         breaksLB=0
#         for example in negData:
#             for v in example:
#                 if u(v)<lb[k]:
#                     breaksLB=1
#                     break
#             if breaksLB==1:
#                 break
#         if breaksLB==0:
#             del lb[k]
#
#         breaksUB = 0
#         for example in negData:
#             for v in example:
#                 if u(v) > ub[k]:
#                     breaksUB = 1
#                     break
#             if breaksUB == 1:
#                 break
#         if breaksUB == 0:
#             del ub[k]
#
#     for i,b in enumerate(binaryExpressions):
#         k=","+str(i)
#         breaksLB=0
#         for example in negData:
#             for (x, y) in pairs(example):
#                 if b.subs({x:x, y: y})<lb[k]:
#                     breaksLB=1
#                     break
#             if breaksLB==1:
#                 break
#         if breaksLB==0:
#             del lb[k]
#
#         breaksUB = 0
#         for example in negData:
#             for (x, y) in pairs(example):
#                 if b.subs({x:x, y: y})>ub[k]:
#                     breaksUB = 1
#                     break
#             if breaksUB == 1:
#                 break
#         if breaksUB == 0:
#             del ub[k]
#
#     for i, u in enumerate(unaryExpressions):
#         for j, b in enumerate(binaryExpressions):
#             k = str(i)+","+str(j)
#             breaksLB = 0
#             for example in negData:
#                 for (x, y) in pairs(example):
#                     if u(b.subs({x:x, y: y})) < lb[k]:
#                         breaksLB = 1
#                         break
#                 if breaksLB == 1:
#                     break
#             if breaksLB == 0:
#                 del lb[k]
#
#             breaksUB = 0
#             for example in negData:
#                 for (x, y) in pairs(example):
#                     if u(b.subs({x:x, y: y})) > ub[k]:
#                         breaksUB = 1
#                         break
#                 if breaksUB == 1:
#                     break
#             if breaksUB == 0:
#                 del ub[k]
#     return lb,ub




if __name__ == '__main__':
    # constraint_learner()
    x, y = symbols('x y')
    for b in generate_expressions(x,y):
        print(b,":", b.subs({x:5, y: 10}))
