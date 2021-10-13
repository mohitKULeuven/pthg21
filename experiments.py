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


def instance_level():
    # import os, glob
    # import pandas as pd
    # path = ""
    # all_files = glob.glob(os.path.join(path, "type*.csv"))
    # df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    # df_merged = pd.concat(df_from_each_file, ignore_index=True)
    # df_merged.to_csv("merged.csv")
    # exit()
    # args = sys.argv[1:]
    for t in [1, 2, 4, 7, 8, 13, 14, 15, 16]:
        csvfile = open(f"type{t:02d}.csv", "w")
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(
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
        for file in files:
            print(file)
            data = json.load(open(file))
            # data = json.load(open(f"instances/type0{args[0]}/instance{args[1]}.json"))
            if data["solutions"]:
                posData = np.array(
                    [np.array(d["list"]).flatten() for d in data["solutions"]]
                )
                negData = np.array(
                    [np.array(d["list"]).flatten() for d in data["nonSolutions"]]
                )
                bounds = learner.constraint_learner(posData, posData.shape[1])
                m, mvars = learner.create_model(data, bounds)
                num_cons = len(m.constraints)
                posDataObj, negDataObj = None, None
                if "objective" in data["solutions"][0]:
                    posDataObj = np.array([d["objective"] for d in data["solutions"]])
                    negDataObj = np.array(
                        [d["objective"] for d in data["nonSolutions"]]
                    )
                m = learner.filter_redundant(m)
                print(
                    f"number of constraints in the model after redundancy check: {len(m.constraints)}"
                )
                perc_pos = learner.check_solutions(m, mvars, posData, max, posDataObj)
                perc_neg = 100 - learner.check_solutions(
                    m, mvars, negData, max, negDataObj
                )
                filewriter.writerow(
                    [
                        t,
                        file,
                        num_cons,
                        len(m.constraints),
                        len(posData),
                        perc_pos,
                        len(negData),
                        perc_neg,
                    ]
                )
    csvfile.close()




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
