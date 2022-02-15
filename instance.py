import itertools
import logging
import random
from functools import reduce

import cpmpy
import numpy as np

import learner

logger = logging.getLogger(__name__)


def nested_map(f, tensor):
    if isinstance(tensor, (list, tuple)):
        return [nested_map(f, st) for st in tensor]
    else:
        return f(tensor)


def load_input_partitions(type_number, input_data):
    if type_number == 1:
        return {
            "edges": [
                [("list", d["nodeA"]), ("list", d["nodeB"])]
                for d in input_data["list"]
            ]
        }
    return {}

def load_input_assignments(type_number, input_data):
    if type_number == 5:
        return {("array", d["row"], d["column"]): d["value"] for d in input_data["preassigned"]}

class Instance:
    def __init__(self, number, json_data, problem_type):
        tensors_lb = {}
        tensors_ub = {}

        self.number = number
        self._cp_vars = None

        self.problem_type = problem_type
        self.jsonSeq = None
        self.input_data = json_data.get("inputData", {})
        self.input_partitions = load_input_partitions(problem_type, self.input_data)
        self.input_assignments = load_input_assignments(problem_type, self.input_data)
        self.constants = {k: v for k, v in self.input_data.items() if isinstance(v, (int, float))}
        if "size" in json_data:
            self.constants["size"] = json_data["size"]

        self.formatTemplate = json_data["formatTemplate"]

        for k, v in json_data["formatTemplate"].items():
            if k != "objective":
                tensors_lb[k] = np.array(nested_map(lambda d: d["low"], v))
                tensors_ub[k] = np.array(nested_map(lambda d: d["high"], v))

        self.tensors_dim = {k: v.shape for k, v in tensors_ub.items()}
        for k, shape in self.tensors_dim.items():
            for i, v in enumerate(shape):
                self.constants[f"{k}_dim{i}"] = v

        self.var_lbs = tensors_lb
        self.var_ubs = tensors_ub

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
            return [
                {_k: np.array(_e[_k]) for _k in self.tensors_dim}
                for _e in _l
            ]

        def import_data__flattened(_l):
            return {
                _k: np.array([np.array(d[_k]).flatten() for d in _l])
                for _k in self.tensors_dim
            }

        self.pos_data = self.neg_data = self.test_data = self.training_data = None

        if json_data["solutions"]:
            self.pos_data = import_data(json_data["solutions"])
            self.neg_data = import_data(json_data["nonSolutions"])
            self.training_data = {
                k: np.array([d[k] for d in self.pos_data])
                for k in self.tensors_dim
            }
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
            # self.inputData = [warehouseCost, customerCost]

        if problem_type == 1:
            inputData = self.input_data["list"]
            lst = []
            for d in inputData:
                lst.append(tuple(sorted(d.values())))
            self.jsonSeq = lst

    @property
    def cp_vars(self):
        if self._cp_vars is None:
            self._cp_vars = dict()
            for k in self.tensors_dim:
                indices = np.array(["-".join(map(str, i)) for i in np.ndindex(*self.tensors_dim[k])])
                index_iterable = np.reshape(np.array(indices), self.tensors_dim[k])
                self._cp_vars[k] = cpmpy.cpm_array(
                    np.vectorize(lambda _i, _lb, _ub: cpmpy.intvar(
                        _lb, _ub, name=f"{k}-{_i}"
                    ))(index_iterable, self.var_lbs[k], self.var_ubs[k])
                )
        return self._cp_vars

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

    def unflatten_data(self, data):
        d = dict()
        offset = 0
        for k, dims in self.tensors_dim.items():
            length = reduce(lambda a, b: a * b, dims)
            d[k] = data[offset:offset + length].reshape(dims)
            offset += length
        return d

    def all_local_indices(self, exp_symbols):
        for name in self.tensors_dim:
            index_pool = [
                (name,) + indices
                for indices in np.ndindex(*self.tensors_dim[name])
            ]
            yield from itertools.combinations(index_pool, len(exp_symbols))

    def example_count(self, positive):
        data = self.pos_data if positive else self.neg_data
        for k in self.tensors_dim:
            return data[k].shape[0]
        raise RuntimeError("Tensor dimensions are empty")

    def learn_model(self, propositional, filter=True, fraction_training=1.0):
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
            mapping=None
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
                m, m_vars, mapping = learner.create_gen_model(
                    var_bounds, expr_bounds, name=k, inputData=self.jsonSeq
                )
            else:
                m, m_vars = learner.create_model(var_bounds, expr_bounds, name=k)

            full_constraints_count += len(m.constraints)

            if filter:
                filtered_bounds[k], constraints = learner.filter_redundant(
                    expr_bounds,
                    m.constraints,
                    mapping
                )
            else:
                filtered_bounds[k], constraints = expr_bounds, m.constraints

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
        if self.problem_type == 3:
            data = self.unflatten_data(data)
            sum = 0
            tmp = np.zeros([len(data["warehouses"]), len(data["customers"])])
            for i, c in enumerate(data["customers"]):
                tmp[c][i] = 1
            sum += np.sum(np.multiply(self.inputData[0], data["warehouses"]))
            sum += np.sum(np.multiply(self.inputData[1], tmp))
            return sum
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
