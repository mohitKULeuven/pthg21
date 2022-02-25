import json
import glob
import csv
import learner
import pickle
import logging
import time
from multiprocessing import Pool
import argparse

from instance import Instance
from learn import learn, create_model
from instances.type06 import model_type06
from instances.nurse_rostering import nurse_rostering_model, nurse_rostering_instance

logger = logging.getLogger(__name__)


def sudoku(training_size):
    print("running Sudoku")
    t = 6
    with open(f"type_{t:02d}_training_size_{training_size}.csv", "w") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=",")
        filewriter.writerow(
            [
                "type",
                "instance",
                "training_size",
                "total_constraints",
                "learned_constraints",
                "learning_time",
                "testing_time",
                "precision",
                "recall",
                # "perc_pos",
                # "perc_neg", "total", "correct_objective", "count"
            ]
        )
        path = f"instances/type{t:02d}/inst*.json"
        files = sorted(glob.glob(path))
        instances = []
        for file in files:
            with open(file) as f:
                instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))
        start = time.time()
        bounding_expressions = learn(instances[:2], training_size)
        learning_time = time.time() - start
        pickleVar = bounding_expressions

        for instance in instances[:3]:
            # len_pos, len_neg = 0, 0
            print(f"instance {instance.number}")
            learned_model, total_constraints = create_model(bounding_expressions, instance, propositional=False)
            print(f"number of constraints: {len(learned_model.constraints)}")
            start_test = time.time()
            precision, recall = learner.compare_models(learned_model, model_type06(instance), instance)
            print(f"precision: {int(precision)}%  |  recall:  {int(recall)}%")

            # perc_pos, perc_neg = None, None
            # if instance.has_solutions():
            #     perc_pos, perc_neg, cnt, co, total = instance.check(learned_model)
            #     print(f"pos: {int(perc_pos)}%  |  neg:  {int(perc_neg)}%")

            filewriter.writerow(
                [
                    t,
                    instance.number,
                    training_size,
                    total_constraints,
                    len(learned_model.constraints),
                    learning_time,
                    time.time() - start_test,
                    precision,
                    recall,
                    # perc_pos,
                    # perc_neg, total, co, cnt
                ]
            )
    pickle.dump(pickleVar, open(f"type{t:02d}_bound_expressions.pickle", "wb"))
    # csvfile.close()


def nurse_rostering(training_size):
    print("running Nurses")
    train_instance1 = nurse_rostering_instance(5, 7)
    train_instance2 = nurse_rostering_instance(6, 10)
    # t = 21
    with open(f"type_nurses_training_size_{training_size}.csv", "w") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=",")
        filewriter.writerow(
            [
                "type",
                "instance",
                "training_size",
                "total_constraints",
                "learned_constraints",
                "learning_time",
                "testing_time",
                "precision",
                "recall",
                # "perc_pos",
                # "perc_neg", "total", "correct_objective", "count"
            ]
        )
        start = time.time()
        bounding_expressions = learn([train_instance1, train_instance2], training_size)
        learning_time = time.time() - start

        test_instance = nurse_rostering_instance(10, 14)

        for instance in [train_instance1, train_instance2, test_instance]:
            print(f"instance {instance.number}")
            learned_model, total_constraints = create_model(bounding_expressions, instance, propositional=False)
            print(f"number of constraints: {len(learned_model.constraints)}")

            start_test = time.time()
            precision, recall = learner.compare_models(learned_model, nurse_rostering_model(instance), instance)
            print(f"precision: {int(precision)}%  |  recall:  {int(recall)}%")

            # perc_pos, perc_neg = None, None
            # if instance.has_solutions():
            #     perc_pos, perc_neg, cnt, co, total = instance.check(learned_model)
            #     print(f"pos: {int(perc_pos)}%  |  neg:  {int(perc_neg)}%")

            filewriter.writerow(
                [
                    "nurses",
                    instance.number,
                    training_size,
                    total_constraints,
                    len(learned_model.constraints),
                    learning_time,
                    time.time() - start_test,
                    precision,
                    recall,
                    # perc_pos,
                    # perc_neg, total, co, cnt
                ]
            )


if __name__ == "__main__":
    # types = [l for l in range(11, 17) if l != 9]
    # types = [int(sys.argv[1])]
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("-s", "--training_size", type=int, nargs='*', default=[1, 5, 10])
    args = parser.parse_args()

    pool = Pool(processes=len(args.training_size))
    if args.exp == "sudoku":
        pool.map(sudoku, args.training_size)
    else:
        pool.map(nurse_rostering, args.training_size)
