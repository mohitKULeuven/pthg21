import random
import sys
import json
import numpy as np
from cpmpy import *
import glob
import csv
import learner
import pickle


def instance_level_generalised(args):
    data = json.load(open(f"instances/type0{args[0]}/instance{args[1]}.json"))
    unseen_data = json.load(open(f"instances/type0{args[0]}/instance{args[2]}.json"))
    if data["solutions"]:
        posData = np.array([np.array(d["list"]).flatten() for d in data["solutions"]])
        negData = np.array(
            [np.array(d["list"]).flatten() for d in data["nonSolutions"]]
        )

        bounds = learner.constraint_learner(posData, posData.shape[1])
        genBounds = learner.generalise_bounds(bounds, posData.shape[1])
        genBounds = learner.filter_trivial(data, genBounds, posData.shape[1])
        # print(genBounds)
        genBounds = learner.filter_redundant(data, genBounds)
        # print("##################")
        print(genBounds)

        mTrain, mvarsTrain, _ = learner.create_gen_model(
            data, genBounds
        )

        mTest, mvarsTest, _ = learner.create_gen_model(
            unseen_data, genBounds
        )
        # print(mTrain)

        posDataObj, negDataObj, unseen_posDataObj, unseen_negDataObj = (
            None,
            None,
            None,
            None,
        )
        if "objective" in data["solutions"][0]:
            posDataObj = np.array([d["objective"] for d in data["solutions"]])
            negDataObj = np.array([d["objective"] for d in data["nonSolutions"]])
            unseen_posDataObj = np.array(
                [d["objective"] for d in unseen_data["solutions"]]
            )
            unseen_negDataObj = np.array(
                [d["objective"] for d in unseen_data["nonSolutions"]]
            )

        #

        unseen_posData = np.array(
            [np.array(d["list"]).flatten() for d in unseen_data["solutions"]]
        )
        unseen_negData = np.array(
            [np.array(d["list"]).flatten() for d in unseen_data["nonSolutions"]]
        )

        perc_pos = learner.check_solutions(mTrain, mvarsTrain, posData, max, posDataObj)
        perc_neg = 100 - learner.check_solutions(
            mTrain, mvarsTrain, negData, max, negDataObj
        )

        perc_unseen_pos = learner.check_solutions(
            mTest, mvarsTest, unseen_posData, max, unseen_posDataObj
        )
        perc_unseen_neg = 100 - learner.check_solutions(
            mTest, mvarsTest, unseen_negData, max, unseen_negDataObj
        )

        print(
            f"{perc_pos}% positives and {perc_neg}% negatives are correctly classified in training"
        )
        print(
            f"{perc_unseen_pos}% positives and {perc_unseen_neg}% negatives are correctly classified in test"
        )


def nested_map(f, tensor):
    if isinstance(tensor, (list, tuple)):
        return [nested_map(f, st) for st in tensor]
    else:
        return f(tensor)


def instance_level():
    for t in range(1, 17):  # [1, 2, 3, 4, 7, 8, 13, 14, 15, 16]:
        with open(f"type{t:02d}.csv", "w") as csv_file:
            file_writer = csv.writer(csv_file, delimiter=",")
            file_writer.writerow(
                [
                    "type",
                    "file",
                    "constraints",
                    "filtered_constraints",
                    "num_pos",
                    "percentage_pos",
                    "num_neg",
                    "percentage_neg",
                ]
            )
            path = f"instances/type{t:02d}/inst*.json"
            files = glob.glob(path)
            for file in sorted(files):
                print(file)
                data = json.load(open(file))

                tensors_lb = {}
                tensors_ub = {}

                for k, v in data["formatTemplate"].items():
                    if k != "objective":
                        tensors_lb[k] = np.array(nested_map(lambda d: d["low"], v))
                        tensors_ub[k] = np.array(nested_map(lambda d: d["high"], v))

                tensors_dim = {k: v.shape for k, v in tensors_ub.items()}
                objective = data["formatTemplate"].get("objective", None)

                if data["solutions"]:
                    full_model, full_model_vars = Model(), []
                    all_constraints_count = 0
                    reduced_constraints_count = 0

                    pos_data = dict()
                    neg_data = dict()

                    pos_data_obj, neg_data_obj = None, None
                    if objective:
                        pos_data_obj = np.array([d["objective"] for d in data["solutions"]])
                        neg_data_obj = np.array(
                            [d["objective"] for d in data["nonSolutions"]]
                        )

                    for k in tensors_dim:
                        pos_data[k] = np.array(
                            [np.array(d[k]).flatten() for d in data["solutions"]]
                        )
                        neg_data[k] = np.array(
                            [np.array(d[k]).flatten() for d in data["nonSolutions"]]
                        )
                        var_bounds = list(zip(tensors_lb[k].flatten(), tensors_ub[k].flatten()))
                        n_pos_examples = pos_data[k].shape[0]
                        # training_indices = random.sample(range(n_pos_examples), int(n_pos_examples * 0.7))
                        training_indices = range(n_pos_examples)
                        expr_bounds = learner.constraint_learner(pos_data[k][training_indices, :], pos_data[k].shape[1])
                        m, mvars = learner.create_model(var_bounds, expr_bounds)
                        full_model_constraint_count = len(m.constraints)
                        all_constraints_count += full_model_constraint_count

                        m = learner.filter_redundant(m)
                        reduced_model_constraint_count = len(m.constraints)
                        print(
                            f"redundancy check [{k}]: {full_model_constraint_count} => {reduced_model_constraint_count}"
                        )
                        reduced_constraints_count += reduced_model_constraint_count
                        full_model += m.constraints
                        full_model_vars += mvars

                    all_pos_data, all_neg_data = None, None
                    for k in tensors_dim:
                        if all_pos_data is None:
                            all_pos_data = pos_data[k]
                            all_neg_data = neg_data[k]
                        else:
                            all_pos_data = np.hstack([all_pos_data, pos_data[k]])
                            all_neg_data = np.hstack([all_neg_data, neg_data[k]])

                    percentage_pos = learner.check_solutions(
                        full_model,
                        cpm_array(full_model_vars),
                        all_pos_data,
                        max,
                        pos_data_obj
                    )

                    percentage_neg = 100 - learner.check_solutions(
                        full_model,
                        cpm_array(full_model_vars),
                        all_neg_data,
                        max,
                        neg_data_obj
                    )

                    print(percentage_pos)
                    print(percentage_neg)

                    file_writer.writerow(
                        [
                            t,
                            file,
                            all_constraints_count,
                            reduced_constraints_count,
                            all_pos_data.shape[0],
                            percentage_pos,
                            all_neg_data.shape[0],
                            percentage_neg,
                        ]
                    )




def type_level(t):
    def common_items(d1, d2):
        result = {}
        for k in d1.keys() & d2.keys():
            v1 = d1[k]
            v2 = d2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                result[k] = common_items(v1, v2)
            elif v1 == v2:
                result[k] = v1
            else:
                if k == 'l':
                    result[k] = min([v1, v2])
                else:
                    result[k] = max([v1, v2])
        return result
    # for t in [1, 2, 4, 7, 8, 13, 14, 15, 16]:
    path = f"instances/type{t:02d}/inst*.json"
    files = glob.glob(path)
    genBoundsList=[]
    for file in files:
        print(file)
        data = json.load(open(file))
        if data["solutions"]:
            posData = np.array([np.array(d["list"]).flatten() for d in data["solutions"]])
            bounds = learner.constraint_learner(posData, posData.shape[1])
            genBounds = learner.generalise_bounds(bounds, posData.shape[1])
            genBounds = learner.filter_trivial(data, genBounds, posData.shape[1])
            genBounds = learner.filter_redundant(data, genBounds)
            genBoundsList.append(genBounds)
    commonBounds=genBoundsList[0]
    for b in genBoundsList[1:]:
        commonBounds = common_items(commonBounds, b)
    return commonBounds

def type_level_experiment():
    csvfile = open(f"type_level_results.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "type",
            "file",
            "constraints",
            "num_pos",
            "percentage_pos",
            "num_neg",
            "percentage_neg",
        ]
    )
    pickleVar={}
    for t in [1, 2, 4, 7, 8, 13, 14, 15, 16]:
        commonBounds = type_level(t)
        pickleVar[t]=commonBounds
        path = f"instances/type{t:02d}/inst*.json"
        files = glob.glob(path)
        for file in files:
            print(file)
            data = json.load(open(file))
            if data["solutions"]:
                posData = np.array(
                    [np.array(d["list"]).flatten() for d in data["solutions"]]
                )
                negData = np.array(
                    [np.array(d["list"]).flatten() for d in data["nonSolutions"]]
                )
                mTrain, mvarsTrain, _ = learner.create_gen_model(
                    data, commonBounds
                )

                posDataObj, negDataObj = (
                    None,
                    None,
                )
                if "objective" in data["solutions"][0]:
                    posDataObj = np.array([d["objective"] for d in data["solutions"]])
                    negDataObj = np.array([d["objective"] for d in data["nonSolutions"]])

                perc_pos = learner.check_solutions(mTrain, mvarsTrain, posData, max, posDataObj)
                perc_neg = 100 - learner.check_solutions(
                    mTrain, mvarsTrain, negData, max, negDataObj
                )
                filewriter.writerow(
                    [
                        t,
                        file,
                        len(mTrain.constraints),
                        len(posData),
                        perc_pos,
                        len(negData),
                        perc_neg,
                    ]
                )
    pickle.dump(pickleVar, open("type_level_models.pickle", "wb"))
    csvfile.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    # instance_level_generalised(args)
    # commonBounds=type_level(int(args[0]))
    type_level_experiment()
