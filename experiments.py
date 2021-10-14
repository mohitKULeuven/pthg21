import random
import sys
import json
from functools import reduce

import numpy as np
import cpmpy
import glob
import csv
import learner
import pickle
import logging
from multiprocessing import Pool

logger = logging.getLogger(__name__)


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

        mTrain, mvarsTrain, _ = learner.create_gen_model(data, genBounds)

        mTest, mvarsTest, _ = learner.create_gen_model(unseen_data, genBounds)
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


class Instance:
    def __init__(self, json_data, problem_type):
        tensors_lb = {}
        tensors_ub = {}

        self.problem_type = problem_type
        self.inputData = None
        self.jsonSeq = None

        for k, v in json_data["formatTemplate"].items():
            if k != "objective":
                tensors_lb[k] = np.array(nested_map(lambda d: d["low"], v))
                tensors_ub[k] = np.array(nested_map(lambda d: d["high"], v))

        self.tensors_dim = {k: v.shape for k, v in tensors_ub.items()}
        self.var_bounds = {
            k: list(zip(tensors_lb[k].flatten(), tensors_ub[k].flatten()))
            for k in self.tensors_dim
        }

        self.objective = json_data["formatTemplate"].get("objective", None)

        def import_objectives(_l):
            return np.array([d["objective"] for d in _l])

        if self.objective:
            self.pos_data_obj = import_objectives(json_data["solutions"])
            self.neg_data_obj = import_objectives(json_data["nonSolutions"])
            self.test_obj = import_objectives(json_data["tests"])
        else:
            self.pos_data_obj = self.neg_data_obj = self.test_obj = None

        def import_data(_l):
            return {
                _k: np.array([np.array(d[_k]).flatten() for d in _l])
                for _k in self.tensors_dim
            }

        self.pos_data = self.neg_data = self.test_data = None

        if json_data["solutions"]:
            self.pos_data = import_data(json_data["solutions"])
            self.neg_data = import_data(json_data["nonSolutions"])
        self.test_data = import_data(json_data["tests"])

        if problem_type == 3:
            inputData = json_data["inputData"]
            customerCost = np.zeros(
                [inputData["nrWarehouses"], inputData["nrCustomers"]]
            )
            for v in inputData["customerCost"]:
                customerCost[v["warehouse"], v["customer"]] = v["cost"]

            warehouseCost = np.zeros(inputData["nrWarehouses"])
            for v in inputData["warehouseCost"]:
                warehouseCost[v["warehouse"]] = v["cost"]
            self.inputData = [warehouseCost, customerCost]

        if problem_type == 1:
            inputData = json_data["inputData"]["list"]
            lst = []
            for d in inputData:
                lst.append(tuple(sorted(d.values())))
            self.jsonSeq = lst

    def has_solutions(self):
        return self.pos_data is not None

    def flatten_data(self, data):
        all_data = None
        for k in self.tensors_dim:
            if all_data is None:
                all_data = data[k]
            else:
                all_data = np.hstack([all_data, data[k]])
        return all_data

    def example_count(self, positive):
        data = self.pos_data if positive else self.neg_data
        for k in self.tensors_dim:
            return data[k].shape[0]
        raise RuntimeError("Tensor dimensions are empty")

    def learn_model(self, propositional, fraction_training=1.0):
        if not self.has_solutions():
            raise AttributeError("Cannot learn from instance without solutions")

        full_model, full_model_vars = cpmpy.Model(), []
        full_constraints_count = 0
        reduced_constraints_count = 0
        filtered_bounds = dict()

        for k in self.tensors_dim:
            pos_data = self.pos_data[k]
            var_bounds = self.var_bounds[k]

            n_pos_examples = pos_data.shape[0]
            if fraction_training == 1.0:
                training_indices = range(n_pos_examples)
            else:
                training_indices = random.sample(
                    range(n_pos_examples), int(n_pos_examples * fraction_training)
                )

            expr_bounds = learner.constraint_learner(
                pos_data[training_indices, :], pos_data.shape[1]
            )
            if not propositional:
                expr_bounds = learner.generalise_bounds(
                    expr_bounds, pos_data.shape[1], self.jsonSeq
                )
                expr_bounds = learner.filter_trivial(
                    var_bounds,
                    expr_bounds,
                    pos_data.shape[1],
                    name=k,
                    inputData=self.jsonSeq,
                )
                m, m_vars, _ = learner.create_gen_model(
                    var_bounds, expr_bounds, name=k, inputData=self.jsonSeq
                )
            else:
                m, m_vars = learner.create_model(var_bounds, expr_bounds, name=k)

            full_constraints_count += len(m.constraints)

            filtered_bounds[k], constraints = learner.filter_redundant(
                var_bounds,
                expr_bounds,
                name=k,
                inputData=self.jsonSeq,
                propositional=propositional,
            )

            reduced_model_constraint_count = len(constraints)
            logger.info(
                f"redundancy check [{k}]: {len(m.constraints)} => {reduced_model_constraint_count}"
            )
            reduced_constraints_count += reduced_model_constraint_count
            full_model += constraints
            full_model_vars += m_vars

        return (
            full_model,
            full_model_vars,
            filtered_bounds,
            dict(
                all_constraints=full_constraints_count,
                reduced_constraints=reduced_constraints_count,
            ),
        )

    def objective_function(self, data):
        return max(data)

    def check(self, model, model_vars):
        percentage_pos = learner.check_solutions(
            model,
            cpmpy.cpm_array(model_vars),
            self.flatten_data(self.pos_data),
            self.objective_function,
            self.pos_data_obj,
        )

        percentage_neg = 100 - learner.check_solutions(
            model,
            cpmpy.cpm_array(model_vars),
            self.flatten_data(self.neg_data),
            max,
            self.neg_data_obj,
        )

        return percentage_pos, percentage_neg

    def test(self, model, model_vars):
        return learner.is_sat(
            model,
            cpmpy.cpm_array(model_vars),
            self.flatten_data(self.test_data),
            max,
            self.test_obj,
        )

    def get_combined_model(self, gen_bounds):
        full_model, full_model_vars = cpmpy.Model(), []
        for k in self.tensors_dim:
            m, m_vars, _ = learner.create_gen_model(
                self.var_bounds[k], gen_bounds[k], name=k, inputData=self.jsonSeq
            )
            full_model += m.constraints
            full_model_vars += m_vars
        return full_model, full_model_vars


def combine_gen_bounds(bounds1, bounds2):
    result = {}
    for k in bounds1.keys() & bounds2.keys():
        v1 = bounds1[k]
        v2 = bounds2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[k] = combine_gen_bounds(v1, v2)
        elif v1 == v2:
            result[k] = v1
        else:
            if k == "l":
                result[k] = min([v1, v2])
            else:
                result[k] = max([v1, v2])
    return result


def nested_map(f, tensor):
    if isinstance(tensor, (list, tuple)):
        return [nested_map(f, st) for st in tensor]
    else:
        return f(tensor)


def instance_level(t):
    pickle_var = {}
    print(f"Type {t}")
    with open(f"type{t:02d}.csv", "w") as csv_file:
        file_writer = csv.writer(csv_file, delimiter=",")
        file_writer.writerow(
            [
                "type",
                "file",
                "constraints",
                "model_used",
                "num_pos",
                "percentage_pos",
                "num_neg",
                "percentage_neg",
            ]
        )
        path = f"instances/type{t:02d}/inst*.json"
        files = sorted(glob.glob(path))

        instances = []
        for file in files:
            with open(file) as f:
                instances.append(Instance(json.load(f), t))

        # Learn propositional models and check their quality
        for i, instance in enumerate(instances):
            if instance.has_solutions():
                m, m_vars, _, stats = instance.learn_model(propositional=True)
                pickle_var[files[i]] = [m, m_vars]
                percentage_pos, percentage_neg = instance.check(m, m_vars)

                # print(f"\tInstance {i}")
                # print("\t\tPropositional")
                # print(*[f"\t\t\t{c}" for c in flatten(m.constraints)], sep="\n")
                # print(f"\t\t\tpos: {int(percentage_pos)}%  |  neg:  {int(percentage_neg)}%")
                # print(f"\t\t\t{instance.test(m, m_vars)}")

                file_writer.writerow(
                    [
                        t,
                        files[i],
                        len(m.constraints),
                        "instance level",
                        len(list(instance.pos_data.values())[0]),
                        percentage_pos,
                        len(list(instance.neg_data.values())[0]),
                        percentage_neg,
                    ]
                )

        gen_bounds = []
        for i, instance in enumerate(instances):
            if instance.has_solutions():
                _, _, filtered_bounds, _ = instance.learn_model(propositional=False)
                gen_bounds.append(filtered_bounds)

        merged_bounds = reduce(combine_gen_bounds, gen_bounds)
        # print(merged_bounds)
        pickle_var[t] = [merged_bounds]
        pickle.dump(pickle_var, open(f"type{t:02d}.pickle", "wb"))

        for i, instance in enumerate(instances):
            # print(f"\tInstance {i}")
            # print("\t\tLifted")
            merged_model, mm_vars = instance.get_combined_model(merged_bounds)
            # print(*[f"\t\t{c}" for c in flatten(merged_model.constraints)], sep="\n")

            if instance.has_solutions():
                percentage_pos, percentage_neg = instance.check(merged_model, mm_vars)
                # print(f"\t\t\tpos: {int(percentage_pos)}%  |  neg:  {int(percentage_neg)}%")
                file_writer.writerow(
                    [
                        t,
                        files[i],
                        len(merged_model.constraints),
                        "type level",
                        len(list(instance.pos_data.values())[0]),
                        percentage_pos,
                        len(list(instance.neg_data.values())[0]),
                        percentage_neg,
                    ]
                )
    csv_file.close()
    # else:
    #     print(f"\t\t\t{instance.test(merged_model, mm_vars)}")


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
                if k == "l":
                    result[k] = min([v1, v2])
                else:
                    result[k] = max([v1, v2])
        return result

    # for t in [1, 2, 4, 7, 8, 13, 14, 15, 16]:
    path = f"instances/type{t:02d}/inst*.json"
    files = glob.glob(path)
    genBoundsList = []
    for file in files:
        print(file)
        data = json.load(open(file))
        if data["solutions"]:
            posData = np.array(
                [np.array(d["list"]).flatten() for d in data["solutions"]]
            )
            bounds = learner.constraint_learner(posData, posData.shape[1])
            genBounds = learner.generalise_bounds(bounds, posData.shape[1])
            genBounds = learner.filter_trivial(data, genBounds, posData.shape[1])
            genBounds = learner.filter_redundant(data, genBounds)
            genBoundsList.append(genBounds)
    commonBounds = genBoundsList[0]
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
    pickleVar = {}
    for t in [1]:
        commonBounds = type_level(t)
        pickleVar[t] = commonBounds
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
                mTrain, mvarsTrain, _ = learner.create_gen_model(data, commonBounds)

                posDataObj, negDataObj = (
                    None,
                    None,
                )
                if "objective" in data["solutions"][0]:
                    posDataObj = np.array([d["objective"] for d in data["solutions"]])
                    negDataObj = np.array(
                        [d["objective"] for d in data["nonSolutions"]]
                    )

                perc_pos = learner.check_solutions(
                    mTrain, mvarsTrain, posData, max, posDataObj
                )
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
    # args = sys.argv[1:]
    # instance_level_generalised(args)
    # commonBounds=type_level(int(args[0]))
    # type_level_experiment()
    t = [1]
    pool = Pool(processes=len(t))
    pool.map(instance_level, t)
