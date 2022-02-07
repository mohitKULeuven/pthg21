import json
import sys

import numpy as np
import cpmpy
import glob
import csv
import learner
import pickle
import logging
import time

from instance import Instance
from learn import learn, create_gen_model
import sys
from instances.type01 import model_type01

logger = logging.getLogger(__name__)


# def instance_level_generalised(args):
#     data = json.load(open(f"instances/type0{args[0]}/instance{args[1]}.json"))
#     unseen_data = json.load(open(f"instances/type0{args[0]}/instance{args[2]}.json"))
#     if data["solutions"]:
#         posData = np.array([np.array(d["list"]).flatten() for d in data["solutions"]])
#         negData = np.array(
#             [np.array(d["list"]).flatten() for d in data["nonSolutions"]]
#         )
#
#         bounds = learner.constraint_learner(posData, posData.shape[1])
#         genBounds = learner.generalise_bounds(bounds, posData.shape[1])
#         genBounds = learner.filter_trivial(data, genBounds, posData.shape[1])
#         # print(genBounds)
#         genBounds = learner.filter_redundant(data, genBounds)
#         # print("##################")
#         print(genBounds)
#
#         mTrain, mvarsTrain, _ = learner.create_gen_model(data, genBounds)
#
#         mTest, mvarsTest, _ = learner.create_gen_model(unseen_data, genBounds)
#         # print(mTrain)
#
#         posDataObj, negDataObj, unseen_posDataObj, unseen_negDataObj = (
#             None,
#             None,
#             None,
#             None,
#         )
#         if "objective" in data["solutions"][0]:
#             posDataObj = np.array([d["objective"] for d in data["solutions"]])
#             negDataObj = np.array([d["objective"] for d in data["nonSolutions"]])
#             unseen_posDataObj = np.array(
#                 [d["objective"] for d in unseen_data["solutions"]]
#             )
#             unseen_negDataObj = np.array(
#                 [d["objective"] for d in unseen_data["nonSolutions"]]
#             )
#
#         #
#
#         unseen_posData = np.array(
#             [np.array(d["list"]).flatten() for d in unseen_data["solutions"]]
#         )
#         unseen_negData = np.array(
#             [np.array(d["list"]).flatten() for d in unseen_data["nonSolutions"]]
#         )
#
#         perc_pos = learner.check_solutions(mTrain, mvarsTrain, posData, max, posDataObj)
#         perc_neg = 100 - learner.check_solutions(
#             mTrain, mvarsTrain, negData, max, negDataObj
#         )
#
#         perc_unseen_pos = learner.check_solutions(
#             mTest, mvarsTest, unseen_posData, max, unseen_posDataObj
#         )
#         perc_unseen_neg = 100 - learner.check_solutions(
#             mTest, mvarsTest, unseen_negData, max, unseen_negDataObj
#         )
#
#         print(
#             f"{perc_pos}% positives and {perc_neg}% negatives are correctly classified in training"
#         )
#         print(
#             f"{perc_unseen_pos}% positives and {perc_unseen_neg}% negatives are correctly classified in test"
#         )


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


def extra_solutions(m, m_vars, existing_sol, solution_limit=20):
    m2 = cpmpy.Model([c for c in m.constraints])
    m2.solve()
    for sol in existing_sol:
        # Tias: this is slow... we should probably use a negative table perhaps
        # though not yet supported in plain CPMpy I think ; )
        m2 += ~cpmpy.all(m_vars == sol)
    solutions = []
    m2 += sum(m_vars) >= 0
    # print(m2)

    # new-style generate all
    from cpmpy_helper import solveAll
    from cpmpy.solvers import CPM_ortools
    s2 = CPM_ortools(m2)

    collector = []
    def myprint():
        print(m_vars.value())
        collector.append(m_vars.value())
    solveAll(s2, display=myprint, solution_limit=solution_limit)

    return collector


def save_results_json(problem_type, instance, tests_classification):
    with open(f"results_type{problem_type}_instance{instance}.json", "w") as f:
        json.dump({"problemType": f"type{problem_type:02d}",
                   "instance": instance,
                   "tests": ["sol" if t else "nonsol" for t in tests_classification]}, f)
    # int(file.split("/")[-1].split(".")[0][8:])


def instance_level(t, filter=False, do_check=False):
    pickle_var = {}
    print(f"Starting type {t}")
    with open(f"type{t:02d}_filter_{filter}.csv", "w") as csv_file:
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
                "time_taken",
                "test_time_taken",
                "number_of_constraints",
                "constraints_after_filter",
            ]
        )
        path = f"instances/type{t:02d}/inst*.json"
        files = sorted(glob.glob(path))

        instances = []
        for file in files:
            with open(file) as f:
                instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

        # Learn propositional models and check their quality
        for i, instance in enumerate(instances):
            if instance.has_solutions():
                start = time.time()
                m, m_vars, _, stats = instance.learn_model(propositional=True, filter=filter)
                time_taken = time.time()-start
                print(f"\tType {t}, {i}: {files[i]}, learned model of {stats['all_constraints']} cons in {time_taken}")
                pickle_var[files[i]] = [m, m_vars]

                if do_check:
                    start = time.time()
                    percentage_pos, percentage_neg = instance.check(m, m_vars)
                    test_time_taken = time.time() - start
                    npos = instance.flatten_data(instance.pos_data).shape[0]
                    nneg = instance.flatten_data(instance.neg_data).shape[0]
                    print(f"\t\tType {t}, {i}: checked {npos+nneg} instances in {test_time_taken}")
                else:
                    percentage_pos, percentage_neg = 0.0, 0.0
                    test_time_taken = 0.0
                # tests_classification = instance.test(m, m_vars)
                # save_results_json(instance.problem_type, instance.number, tests_classification)

                # all_data = np.vstack(
                #     [instance.flatten_data(instance.pos_data), instance.flatten_data(instance.neg_data),
                #      instance.flatten_data(instance.test_data)])
                # print(m.constraints)
                # print([(v, v.value()) for v in m_vars])
                # flat_extra = extra_solutions(m, cpmpy.cpm_array(m_vars), all_data)
                # extra = [instance.unflatten_data(d) for d in flat_extra]
                #
                # if instance.objective:
                #     objectives = [instance.objective_function(d) for d in flat_extra]
                #     for objective in objectives:
                #         extra["objective"] = objective
                # output_dictionary[i]["extras"] = extra

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
                        time_taken,
                        test_time_taken,
                        stats["all_constraints"],
                        stats["reduced_constraints"],
                    ]
                )


        # gen_bounds = []
        # for i, instance in enumerate(instances):
        #     if instance.has_solutions():
        #         _, _, filtered_bounds, _ = instance.learn_model(propositional=False, filter=filter)
        #         gen_bounds.append(filtered_bounds)
        #
        # merged_bounds = reduce(combine_gen_bounds, gen_bounds)
        # # print(merged_bounds)
        # pickle_var[t] = [merged_bounds]
        #
        # for i, instance in enumerate(instances):
        #     # print(f"\tInstance {i}")
        #     # print("\t\tLifted")
        #     merged_model, mm_vars = instance.get_combined_model(merged_bounds)
        #     # print(*[f"\t\t{c}" for c in flatten(merged_model.constraints)], sep="\n")
        #
        #     if instance.has_solutions():
        #         percentage_pos, percentage_neg = instance.check(merged_model, mm_vars)
        #         # print(f"\t\t\tpos: {int(percentage_pos)}%  |  neg:  {int(percentage_neg)}%")
        #         file_writer.writerow(
        #             [
        #                 t,
        #                 files[i],
        #                 len(merged_model.constraints),
        #                 "type level",
        #                 len(list(instance.pos_data.values())[0]),
        #                 percentage_pos,
        #                 len(list(instance.neg_data.values())[0]),
        #                 percentage_neg,
        #             ]
        #         )
        #     else:
        #         tests_classification = instance.test(merged_model, mm_vars)
        #         save_results_json(instance.problem_type, instance.number, tests_classification)

                # flat_extra = extra_solutions(m, cpmpy.cpm_array(m_vars), instance.flatten_data(instance.test_data))
                # extra = [instance.unflatten_data(d) for d in flat_extra]
                #
                # if instance.objective:
                #     objectives = [instance.objective_function(d) for d in flat_extra]
                #     for objective in objectives:
                #         extra["objective"] = objective
                #
                # output_dictionary[i]["extras"] = extra

    csv_file.close()
    with open(f"pickled_models_type{t:02d}_filter_{filter}.pickle") as pickle_file:
        pickle.dump(pickle_var, pickle_file)


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


def generalized_learning_experiment(types):
    csvfile = open(f"type_level_results.csv", "w")
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "type",
            "instance",
            "total_constraints",
            "learned_constraints",
            "learning_time",
            "testing_time",
            "recall",
            "precision"
        ]
    )
    pickleVar = {}
    for t in types:
        print(f"type{t}")
        with open(f"type {t:02d}.csv", "w") as csv_file:
            path = f"instances/type{t:02d}/inst*.json"
            files = sorted(glob.glob(path))
            instances = []
            for file in files:
                with open(file) as f:
                    instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))
            start = time.time()
            bounding_expressions = learn(instances)
            learning_time = time.time() - start
            pickleVar[t] = bounding_expressions

            for instance in instances[:1]:
                # len_pos, len_neg = 0, 0
                print(f"instance {instance.number}")
                learned_model, total_constraints = create_gen_model(bounding_expressions, instance)
                start_test = time.time()
                precision, recall = learner.compare_models(learned_model, model_type01(instance), instance)
                # recall = cc*100/tc
                # precision = cc*100/lc
                # print(recall, precision)
                # perc_pos, perc_neg = None, None
                # if instance.pos_data is not None:
                #     len_pos = len(instance.pos_data)
                #     perc_pos = learner.check_solutions_fast(
                #         learned_model, instance.cp_vars, instance.pos_data, max, instance.pos_data_obj
                #     )
                #     print("perc_pos: ", perc_pos)
                # if instance.neg_data is not None:
                #     len_neg = len(instance.neg_data)
                #     perc_neg = 100 - learner.check_solutions_fast(
                #         learned_model, instance.cp_vars, instance.neg_data, max, instance.neg_data_obj
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
                        recall,
                        precision
                    ]
                )
    pickle.dump(pickleVar, open("type_level_models.pickle", "wb"))
    csvfile.close()


if __name__ == "__main__":
    # args = sys.argv[1:]
    # instance_level_generalised(args)
    # commonBounds=type_level(int(args[0]))
    # type_level_experiment()
    # print(instance_level(t[0]))
    # t = [l for l in range(1, 17) if l != 9]
    #
    # pool = Pool(processes=len(t))
    # results = pool.map(instance_level, t)
    # print(results)

    # types = [l for l in range(1, 17) if l != 9]
    types = [int(sys.argv[1])]
    # print(types)
    generalized_learning_experiment(types)

    # for t in types:
    #     with open(f"type{t:02d}.csv", "w") as csv_file:
    #         path = f"instances/type{t:02d}/inst*.json"
    #         files = sorted(glob.glob(path))
    #         instances = []
    #         for file in files:
    #             with open(file) as f:
    #                 instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))
    #         learn(instances)

