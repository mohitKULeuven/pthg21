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
from learn import learn, create_model, learn_propositional
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
from instances.type11 import model_type11
from instances.type12 import model_type12
from instances.type13 import model_type13
from instances.type14 import model_type14
from instances.type15 import model_type15
from instances.type16 import model_type16

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
    elif t == 11:
        return model_type11(instance)
    elif t == 12:
        return model_type12(instance)
    elif t == 13:
        return model_type13(instance)
    elif t == 14:
        return model_type14(instance)
    elif t == 15:
        return model_type15(instance)
    elif t == 16:
        return model_type16(instance)



def propositional_level_experiment(t):
    with open(f"type_{t:02d}_propositional.csv", "w") as csv_file:
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
                "perc_neg", "total", "correct_objective", "count"
            ]
        )
        path = f"instances/type{t:02d}/inst*.json"
        files = sorted(glob.glob(path))
        instances = []
        for file in files:
            with open(file) as f:
                instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

        for instance in instances:
            print(f"instance {instance.number}")
            if not instance.has_solutions():
                continue
            start = time.time()
            bounding_expressions = learn_propositional(instance)
            learning_time = time.time() - start
            pickleVar = bounding_expressions
            learned_model, total_constraints = create_model(bounding_expressions, instance, propositional=True)

            start_test = time.time()
            precision, recall = learner.compare_models(learned_model, true_model(t, instance), instance)
            print(f"precision: {int(precision)}%  |  recall:  {int(recall)}%")

            perc_pos, perc_neg = None, None
            if instance.has_solutions():
                perc_pos, perc_neg, cnt, co, total = instance.check(learned_model)
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
                    perc_neg, total, co, cnt
                ]
            )
    pickle.dump(pickleVar, open(f"type{t:02d}_bound_expressions.pickle", "wb"))
    # csvfile.close()


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
                "perc_neg", "total", "correct_objective", "count"
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
            learned_model, total_constraints = create_model(bounding_expressions, instance, propositional=False)
            start_test = time.time()

            precision, recall = learner.compare_models(learned_model, true_model(t, instance), instance)
            print(f"precision: {int(precision)}%  |  recall:  {int(recall)}%")

            perc_pos, perc_neg = None, None
            if instance.has_solutions():
                perc_pos, perc_neg, cnt, co, total = instance.check(learned_model)
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
                    perc_neg, total, co, cnt
                ]
            )
    pickle.dump(pickleVar, open(f"type{t:02d}_bound_expressions.pickle", "wb"))
    # csvfile.close()


def exp_learn_specific_model(t):
    path = f"instances/type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    return learn(instances)


if __name__ == "__main__":
    # types = [l for l in range(11, 17) if l != 9]
    types = [int(sys.argv[1])]
    pool = Pool(processes=len(types))
    if sys.argv[2] == "propositional":
        pool.map(propositional_level_experiment, types)
    else:
        pool.map(generalized_learning_experiment, types)