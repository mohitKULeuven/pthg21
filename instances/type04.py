import glob
import json

import numpy as np
from cpmpy import *
import learner
from learn import learn, create_gen_model
from instance import Instance

"""
    Graph coloring

    Arguments:
    - formatTemplate: dict as in the challenge, containing:
        "list": list of dicts of 'high', 'low', 'type'
    - inputData: dict as in challenge, containing:
        "list": list of dicts with 'nodeA', 'nodeB'
"""


def get_model(param):
    m = param['marks']
    n = 2 ** (m - 1)

    marks = VarArray(m, n, 'm')
    distance = [marks[i] - marks[j] for i in range(1, m) for j in range(i)]

    model = Model(
        Minimise(marks[-1]),  # objective function

        [marks[i - 1] < marks[i] for i in range(1, m)],
        AllDiff(distance),
        marks[0] == 0,  # symmetry breaking

        [distance[i * (i - 1) / 2 + j] >= ruler[i - j] for i in range(1, m) for j in range(0, i - 1) if (i - j < m)]
    )

    return marks, model
def model_type04(instance: Instance):
    marks = instance.cp_vars['list']
    l = len(marks)

    model = Model()

    model += marks[0] == 0  # symm breaking
    model += [marks[i] < marks[i+1] for i in range(l-1)]

    diffs = [marks[i] - marks[j] for i in range(l) for j in range(i+1,l)]
    model += AllDifferent(diffs)

    #model.minimize(max(marks)) # or marks[l-1], as you wish

    return model

if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 4
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (Golomb ruler)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type04(inst)
    print(m)

    # sanity check ground truth
    print("pos/neg accuracy", inst.check(m))

