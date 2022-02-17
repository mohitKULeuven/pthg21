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
from instances.type02 import model_type02
from instances.type03 import model_type03
from instances.type04 import model_type04
from instances.type05 import model_type05
from instances.type06 import model_type06
from instances.type07 import model_type07
from instances.type08 import model_type08
from instances.type10 import model_type10

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

def true_model(t, instance):
    if t == 1:
        return model_type01(instance)
    elif t == 2:
        return model_type02(instance)
    elif t == 3:
        return model_type03(instance)
    elif t == 4:
        return model_type04(instance)
    elif t == 5:
        return model_type05(instance)
    elif t == 6:
        return model_type06(instance)
    elif t == 7:
        return model_type07(instance)
    elif t == 8:
        return model_type08(instance)
    elif t == 10:
        return model_type10(instance)


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
                "precision",
                "recall",
                "perc_pos",
                "perc_neg",
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

            precision, recall = learner.compare_models(learned_model, true_model(t, instance), instance)
            print(f"precision: {int(precision)}%  |  recall:  {int(recall)}%")

            perc_pos, perc_neg = None, None
            if instance.has_solutions():
                perc_pos, perc_neg = instance.check(learned_model)
                print(f"pos: {int(perc_pos)}%  |  neg:  {int(perc_neg)}%")

            filewriter.writerow(
                [
                    t,
                    instance.number,
                    total_constraints,
                    len(learned_model.constraints),
                    learning_time,
                    time.time() - start_test,
                    precision,
                    recall,
                    perc_pos,
                    perc_neg,
                ]
            )
    pickle.dump(pickleVar, open(f"type{t:02d}_bound_expressions.pickle", "wb"))
    # csvfile.close()


if __name__ == "__main__":
    # types = [l for l in range(1, 17) if l != 9]
    types = [int(sys.argv[1])]
    pool = Pool(processes=1)
    pool.map(generalized_learning_experiment, types)