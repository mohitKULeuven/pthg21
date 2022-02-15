from collections import defaultdict

import sympy
from sympy import symbols, lambdify, sympify, Symbol

from instance import Instance
import itertools as it
import numpy as np
import enum


# def unary_operators():
#     def modulo(x):
#         return x % 2
#
#     def power(x):
#         return x * x
#
#     def identity(x):
#         return x
#
#     for f in [identity, abs]:
#         yield f


def generate_unary_exp(x):
    yield x
    # yield abs(x) for negative data


def generate_binary_expr(x, y):
    yield x + y
    yield x - y
    yield abs(x - y)


# model = []
# for expr in unary_grammar:
#     1. make expression
#     1b ?? check for (syntactic) equivalence with previous expressions?
#     for each inst:
#         2. compute bounds over all vars
#         3. search for grouped constraints over these bounds of this expression
#     4a: symbolize bounds across instances and only keep general ones
#     4. remove trivial bounds/constraints (both general and remaining single)
#     5. check that the found constraints are not implied by current model
#     6. add remaining to model
# same for expr in binary_grammar:

class PartitionType(enum.Enum):
    row = "row", 0
    column = "column", 1
    full = "full", None


class Partition:
    def generate_partition_indices(self, instance: Instance):
        raise NotImplementedError()


class GenericPartition(Partition):
    def __init__(self, tensor_name, partition_type):
        self.tensor_name = tensor_name
        self.partition_type = partition_type

    def generate_partition_indices(self, instance: Instance):
        index_pool = []
        if self.partition_type.value[1] is not None:
            for i in range(instance.tensors_dim[self.tensor_name][self.partition_type.value[1]]):
                indices = [
                    (self.tensor_name,) + indices
                    for indices in np.ndindex(*instance.tensors_dim[self.tensor_name])
                    if indices[self.partition_type.value[1]] == i
                ]
                index_pool.append(indices)
        else:
            indices = [
                (self.tensor_name,) + indices
                for indices in np.ndindex(*instance.tensors_dim[self.tensor_name])
            ]
            index_pool.append(indices)

        return index_pool

    def __repr__(self):
        return f"Partition({self.tensor_name}, {self.partition_type.value[0]})"


class InputPartition(Partition):
    def __init__(self, input_key):
        self.input_key = input_key

    def generate_partition_indices(self, instance: Instance):
        return instance.input_partitions[self.input_key]

    def __repr__(self):
        return f"InputPartition({self.input_key})"



def gen_partitions(type_shape: dict[str, int], input_keys: list[str]) -> list[Partition]:
    """
    Generates partitions for a given shape
    :param type_shape: Maps tensor names to the number of dimensions they have
    :param input_keys: The keys of (custom) input partitions provided to the learning algorithms
    :return:  A list of partitions
    """
    partitions = []
    for name, n_dims in type_shape.items():
        partitions.append(GenericPartition(name, PartitionType.full))
        if n_dims == 2:
            partitions.append(GenericPartition(name, PartitionType.row))
            partitions.append(GenericPartition(name, PartitionType.column))
        elif n_dims > 2:
            raise NotImplementedError()
    for key in input_keys:
        partitions.append(InputPartition(key))
    return partitions


class Sequence(enum.Enum):
    ALL_UNARY = "all_unary", 1
    EVEN_UNARY = "even_unary", 1
    ODD_UNARY = "odd_unary", 1

    ALL_PAIRS = "all_pairs", 2
    SEQUENCE_PAIRS = "sequence_pairs", 2


def gen_sequences(exp_symbols):
    return [
        s
        for s in Sequence
        if (s.value[1] == len(exp_symbols)) and s not in (Sequence.EVEN_UNARY, Sequence.ODD_UNARY)
    ]


def gen_index_groups(sequence, partition_indices):
    # print(partition_indices)
    if sequence == Sequence.ALL_UNARY:
        return [(i,) for i in partition_indices]
    if sequence == Sequence.EVEN_UNARY:
        return [(partition_indices[i],) for i in range(0, len(partition_indices), 2)]
    if sequence == Sequence.ODD_UNARY:
        return [(partition_indices[i],) for i in range(1, len(partition_indices), 2)]

    if sequence == Sequence.ALL_PAIRS:
        return list(it.combinations(partition_indices, r=2))
    if sequence == Sequence.SEQUENCE_PAIRS:
        return [(partition_indices[i], partition_indices[i + 1]) for i in range(len(partition_indices) - 1)]


def filter_partition_bounds(partition_bounds, threshold=1):
    # cv = lambda x: np.std(x, ddof=1) / np.mean(x)
    #
    # for sequence in all_sequences:
    #     lbs = [lb for lb, _ in partition_bounds[sequence]]
    #     ubs = [ub for _, ub in partition_bounds[sequence]]
    #
    #     if cv(lbs) > threshold:
    #         partition_bounds[sequence]

    return partition_bounds  # Filter using STD-DEV


def compute_shape(instances: list[Instance]):
    return {key: len(shape) for key, shape in instances[0].tensors_dim.items()}


def compute_input_keys(instances: list[Instance]):
    return [key for key in instances[0].input_partitions]


FeatureName = str
InstanceNumber = Value = int


def compute_candidate_features(instances: list[Instance]) -> dict[FeatureName, dict[InstanceNumber, Value]]:
    return {k: {instance.number: instance.constants[k] for instance in instances} for k in instances[0].constants}


def fit_feature_expressions(
        bounds: dict[InstanceNumber, tuple],
        candidate_features: dict[FeatureName, dict[InstanceNumber, Value]],
        threshold=1
):
    # print(bounds, candidate_features)
    # print(bounds)
    lb = min([bounds[instance_number][0] for instance_number in bounds])
    ub = max([bounds[instance_number][1] for instance_number in bounds])
    min_error_lb = [bounds[instance_number][0] - lb for instance_number in bounds]
    min_error_ub = [ub - bounds[instance_number][1] for instance_number in bounds]

    for f, values in candidate_features.items():
        bias_lb = min(
            [bounds[instance_number][0] - values[instance_number]
             for instance_number in bounds]
        )
        error = [bounds[instance_number][0] - values[instance_number] - bias_lb for instance_number in bounds]
        if sum(error) < sum(min_error_lb):
            min_error_lb = error
            lb = bias_lb + sympy.S(f)

        bias_ub = max(
            [bounds[instance_number][1] - values[instance_number]
             for instance_number in bounds]
        )
        error = [values[instance_number] + bias_ub - bounds[instance_number][1] for instance_number in bounds]
        if sum(error) < sum(min_error_ub):
            min_error_ub = error
            ub = bias_ub + sympy.S(f)

    cv = lambda x: sum(x) / len(x)
    if cv(min_error_lb) > threshold:
        lb = None
    if cv(min_error_ub) > threshold:
        ub = None
    # print(min_error_lb, min_error_ub)
    return lb, ub


def learn_for_expression(instances: list[Instance], expression, exp_symbols):
    name = str(expression)
    f = lambdify(exp_symbols, expression, "math")
    bounds_over_partitions_across_instances = dict()
    type_shape = compute_shape(instances)
    input_keys = compute_input_keys(instances)
    candidate_features = compute_candidate_features(instances)
    #
    all_partitions = gen_partitions(type_shape, input_keys)
    all_sequences = gen_sequences(exp_symbols)

    # print("expression", expression)

    for instance in instances:
        if not instance.has_solutions():
            continue

        local_bounds = dict()

        for indices in instance.all_local_indices(exp_symbols):
            vals = f(*[instance.training_data[ind[0]][(slice(None),) + ind[1:]] for ind in indices])
            # print("values across training data", vals)
            local_bounds[indices] = min(vals), max(vals)
            # print("learned bounds", local_bounds[indices])
            # print()

        bounds_over_partitions = dict()

        # columns of a matrix, rows of a matrix, all values of a list
        for partitions in all_partitions:
            partition_bounds = defaultdict(list)

            # all indices in a specific column
            for partition_indices in partitions.generate_partition_indices(instance):
                # print(partitions, partition_indices)

                for sequence in all_sequences:  # all-pairs, sequential values
                    # print(sequence)

                    # for index_group in gen_index_groups(sequence, partition_indices):
                    # print("\t", index_group)

                    partition_sequence_bounds = [
                        local_bounds[index_group]
                        for index_group in gen_index_groups(sequence, partition_indices)
                        # one pair of a specific column
                    ]

                    # print(partition_sequence_bounds)
                    partition_bounds[sequence].append((
                        min([lb for lb, _ in partition_sequence_bounds]),
                        max([ub for _, ub in partition_sequence_bounds])
                    ))
                    # print(sequence, partition_bounds[sequence])

            partition_bounds = filter_partition_bounds(partition_bounds)

            bounds_over_partitions[partitions] = {
                seq:
                    (min([lb for lb, _ in partition_bounds[seq]]), max([ub for _, ub in partition_bounds[seq]]))
                for seq in all_sequences
            }

            # print(expression, partitions, bounds_over_partitions[partitions])

        bounds_over_partitions_across_instances[instance.number] = bounds_over_partitions

    # Symbolic

    bounding_expressions = dict()

    # print([i for i, ins in enumerate(instances) if ins.has_solutions()])

    for partitions in all_partitions:
        for sequence in all_sequences:
            bounds = {instance.number:
                          bounds_over_partitions_across_instances[instance.number][partitions][sequence]
                      for instance in instances if instance.has_solutions()}
            # print(bounds)
            bounding_expressions[(partitions, sequence)] = fit_feature_expressions(bounds, candidate_features)
            # print("\t", partitions, sequence, bounding_expressions[(partitions, sequence)])

    # print(bounding_expressions)
    return bounding_expressions


def learn(instances):
    x, y = symbols("x y")
    bounding_expressions = dict()
    for u in generate_unary_exp(x):
        for key, val in learn_for_expression(instances, u, [x]).items():
            bounding_expressions[(u,) + key] = val

    for b in generate_binary_expr(x, y):
        for key, val in learn_for_expression(instances, b, [x, y]).items():
            bounding_expressions[(b,) + key] = val

    return bounding_expressions

    # full_model, full_model_vars = cpmpy.Model(), []
    # full_constraints_count = 0
    # reduced_constraints_count = 0
    # filtered_bounds = dict()
    #
    # for k in self.tensors_dim:
    #     pos_data = self.pos_data[k]
    #     var_bounds = self.var_bounds[k]
    #
    #     n_pos_examples = pos_data.shape[0]
    #     if fraction_training == 1.0:
    #         training_indices = range(n_pos_examples)
    #     else:
    #         training_indices = random.sample(
    #             range(n_pos_examples), int(n_pos_examples * fraction_training)
    #         )
    #
    #     expr_bounds = learner.constraint_learner(
    #         pos_data[training_indices, :], pos_data.shape[1]
    #     )
    #     mapping = None
    #     if not propositional:
    #         expr_bounds = learner.generalise_bounds(
    #             expr_bounds, pos_data.shape[1], self.jsonSeq
    #         )
    #         expr_bounds = learner.filter_trivial(
    #             var_bounds,
    #             expr_bounds,
    #             pos_data.shape[1],
    #             name=k,
    #             inputData=self.jsonSeq,
    #         )
    #         m, m_vars, mapping = learner.create_gen_model(
    #             var_bounds, expr_bounds, name=k, inputData=self.jsonSeq
    #         )
    #     else:
    #         m, m_vars = learner.create_model(var_bounds, expr_bounds, name=k)
    #
    #     full_constraints_count += len(m.constraints)
    #
    #     if filter:
    #         filtered_bounds[k], constraints = learner.filter_redundant(
    #             expr_bounds,
    #             m.constraints,
    #             mapping
    #         )
    #     else:
    #         filtered_bounds[k], constraints = expr_bounds, m.constraints
    #
    #     reduced_model_constraint_count = len(constraints)
    #     logger.info(
    #         f"redundancy check [{k}]: {len(m.constraints)} => {reduced_model_constraint_count}"
    #     )
    #     reduced_constraints_count += reduced_model_constraint_count
    #     full_model += constraints
    #     full_model_vars += m_vars
    #
    # return (
    #     full_model,
    #     full_model_vars,
    #     filtered_bounds,
    #     dict(
    #         all_constraints=full_constraints_count,
    #         reduced_constraints=reduced_constraints_count,
    #     ),
    # )


from cpmpy import *


def create_gen_model(general_bounds, instance: Instance):
    # cp_vars = instance.cp_vars
    exp_symbols = symbols("x y")

    def ground_bound(_bound):
        try:
            for k, v in instance.constants.items():
                _bound = _bound.subs(Symbol(k), v)
            return int(_bound)
        except AttributeError:
            return _bound

    m = Model()
    total_constraints = 0
    for (expr, partitions, sequences), (lb, ub) in general_bounds.items():
        # if lb is None and ub is None:
        #     continue
        expr = sympify(expr)
        for partition_indices in partitions.generate_partition_indices(instance):
            for indices in gen_index_groups(sequences, partition_indices):
                cp_vars = [instance.cp_vars[index[0]][index[1:]] for index in indices]
                f = lambdify(exp_symbols[:len(cp_vars)], expr, "math")
                cpm_e = f(*cp_vars)
                if lb is not None:
                    m += [cpm_e >= ground_bound(lb)]
                if ub is not None:
                    m += [cpm_e <= ground_bound(ub)]
                total_constraints += 2

    if instance.input_assignments:
        for k, v in instance.input_assignments.items():
            m += [instance.cp_vars[k[0]][k[1:]] == v]
    return m, total_constraints
