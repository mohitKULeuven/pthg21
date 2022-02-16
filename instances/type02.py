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


def model_type02(instance: Instance):
    queens = instance.cp_vars['list']
    N = len(queens)

    model = Model(
        AllDifferent(queens),
        AllDifferent([queens[i] + i for i in range(N)]),
        AllDifferent([queens[i] - i for i in range(N)])
    )
    return model

if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 2
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (n-queens)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type02(inst)
    print(m)


    # sanity check ground truth
    sols, non_sols = [], []
    if inst.pos_data is not None:
        sols = [np.hstack([list(d[k].flatten()) for k in inst.tensors_dim]) for d in inst.pos_data]
        perc_pos = learner.check_solutions_fast(
            m, inst.cp_vars, sols, max, inst.pos_data_obj
        )
        print("perc_pos: ", perc_pos)
    if inst.neg_data is not None:
        non_sols = [np.hstack([list(d[k].flatten()) for k in inst.tensors_dim]) for d in inst.neg_data]
        perc_neg = 100 - learner.check_solutions_fast(
            m, inst.cp_vars, non_sols, max, inst.neg_data_obj
        )
        print("perc_neg: ", perc_neg)

