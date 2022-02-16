import json
import numpy as np
import cpmpy
import glob
import csv
import learner
import pickle
import logging
import time
from multiprocessing import Pool

from instance import Instance
from learn import learn, create_gen_model
import sys
from instances.type01 import model_type01

logger = logging.getLogger(__name__)


def flatten(l):
    result = []

    def flatten_rec(_l):
        if isinstance(_l, (list, tuple)):
            for e in _l:
                flatten_rec(e)
        else:
            result.append(_l)

    flatten_rec(l)
    return result


def generalized_learning_experiment(t):
    print(f"type{t}")
    with open(f"type {t:02d}.csv", "w") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=",")
        filewriter.writerow(
            [
                "type",
                "instance",
                "total_constraints",
                "learned_constraints",
                "learning_time",
                "testing_time",
                "perc_pos",
                "perc_neg"
            ]
        )
        path = f"instances/type{t:02d}/inst*.json"
        files = sorted(glob.glob(path))
        instances = []
        for file in files:
            with open(file) as f:
                instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))
        start = time.time()
        bounding_expressions = learn(instances)
        learning_time = time.time() - start
        pickleVar = bounding_expressions

        for instance in instances:
            # len_pos, len_neg = 0, 0
            print(f"instance {instance.number}")
            learned_model, total_constraints = create_gen_model(bounding_expressions, instance)
            start_test = time.time()
            # precision, recall = learner.compare_models(learned_model, model_type01(instance), instance)
            # recall = cc*100/tc
            # precision = cc*100/lc
            # print(recall, precision)
            perc_pos, perc_neg = None, None
            if instance.has_solutions():
                perc_pos, perc_neg = instance.check(learned_model)
                print(f"pos: {int(perc_pos)}%  |  neg:  {int(perc_neg)}%")
            #     sols = [np.hstack([list(d[k].flatten()) for k in instance.tensors_dim]) for d in instance.pos_data]
            #     pp = learner.check_solutions(learned_model, np.hstack([instance.cp_vars[k].flatten() for k in instance.cp_vars]),
            #                                  sols, max, objectives=None)
            #     print("percentage_positive: ", pp)
            #     perc_pos = learner.check_solutions_fast(
            #         learned_model, instance.cp_vars, sols, max, instance.pos_data_obj
            #     )
            #     print("perc_pos: ", perc_pos)
            # if instance.neg_data is not None:
            #     non_sols = [np.hstack([list(d[k].flatten()) for k in instance.tensors_dim]) for d in instance.neg_data]
            #     perc_neg = 100 - learner.check_solutions_fast(
            #         learned_model, instance.cp_vars, non_sols, max, instance.neg_data_obj
            #     )
            #     print("perc_neg: ", perc_neg)
            filewriter.writerow(
                [
                    t,
                    instance.number,
                    total_constraints,
                    len(learned_model.constraints),
                    learning_time,
                    time.time() - start_test,
                    perc_pos,
                    perc_neg,
                    # recall,
                    # precision,
                ]
            )
    pickle.dump(pickleVar, open(f"type{t:02d}_bound_expressions.pickle", "wb"))
    # csvfile.close()


if __name__ == "__main__":
    # types = [l for l in range(1, 17) if l != 9]
    types = [int(sys.argv[1])]
    pool = Pool(processes=1)
    pool.map(generalized_learning_experiment, types)