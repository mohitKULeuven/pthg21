from collections import defaultdict

from sympy import symbols, lambdify, sympify, Symbol

from instance import Instance


def unary_operators():
    def modulo(x):
        return x % 2

    def power(x):
        return x * x

    def identity(x):
        return x

    for f in [identity, abs]:
        yield f


def generate_unary_exp(x):
    yield x
    # for u in unary_operators():
    #     yield u(x)


def binary_operators(x, y):
    # for u in unary_operators():
    yield x + y
    yield x - y
    yield y - x


def generate_binary_expr(x, y):
    yield x + y
    yield x - y
    yield y - x
    yield abs(y-x)
    yield abs(x-y)


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


def gen_partitions(shape):
    # Partition objects
    raise NotImplementedError()  # TODO


def gen_sequences(exp_symbols):
    # Sequence objects
    raise NotImplementedError()  # TODO


def filter_partition_bounds(partition_bounds):
    return partition_bounds  # Filter using STD-DEV


def compute_shape(instances: list[Instance]):
    raise NotImplementedError()  # TODO


def compute_candidate_features(instances: list[Instance]):
    raise NotImplementedError()  # TODO


FeatureName = str
InstanceNumber = Value = int


def fit_feature_expressions(bounds: dict[InstanceNumber, tuple], candidate_features: dict[FeatureName, dict[InstanceNumber, Value]]):
    raise NotImplementedError()  # TODO


def learn_for_expression(instances: list[Instance], expression, exp_symbols):
    name = str(expression)
    f = lambdify(exp_symbols, expression, "math")
    bounds_over_partitions_across_instances = dict()
    # shape = compute_shape(instances)
    # candidate_features = compute_candidate_features(instances)
    #
    # all_partitions = gen_partitions(shape)
    # all_sequences = gen_sequences(exp_symbols)

    print("expression", expression)

    for instance in instances:
        if not instance.has_solutions():
            continue

        print("instance", instance.number, "expression", expression)

        local_bounds = dict()

        for indices in instance.all_local_indices(exp_symbols):
            print("index", indices)
            vals = f(*[instance.pos_data[ind[0]][(slice(None),) + ind[1:]] for ind in indices])
            # print("values across training data", vals)
            local_bounds[indices] = min(vals), max(vals)
            print("learned bounds", local_bounds[indices])
            print()

        bounds_over_partitions = dict()

    #     for partitions in all_partitions:  # columns of a matrix, rows of a matrix, all values of a list
    #         partition_bounds = defaultdict(list)
    #         for partition in partitions:  # all indices in a specific column
    #             for sequence in all_sequences:  # all-pairs, sequential values
    #                 partition_sequence_bounds = [
    #                     local_bounds[index_group]
    #                     for index_group in sequence.generate_sequences(partition)  # TODO
    #                     # one pair of a specific column
    #                 ]
    #                 partition_bounds[sequence].append((
    #                     min(lb for lb, _ in partition_sequence_bounds),
    #                     max(ub for _, ub in partition_sequence_bounds)
    #                 ))
    #
    #         partition_bounds = filter_partition_bounds(partition_bounds)
    #         bounds_over_partitions[partitions] = (
    #             {seq: min(lb for lb, _ in partition_bounds[seq]) for seq in all_sequences},
    #             {seq: max(ub for _, ub in partition_bounds[seq]) for seq in all_sequences}
    #         )
    #
    #     bounds_over_partitions_across_instances[instance.number] = bounds_over_partitions
    #
    # # Symbolic
    #
    # bounding_expressions = dict()
    #
    # for partitions in all_partitions:
    #     for sequence in all_sequences:
    #         bounds = {
    #             instance.number:
    #             bounds_over_partitions_across_instances[instance.number][partitions][sequence]
    #             for instance in instances
    #         }
    #         bounding_expressions[(partitions, sequence)] = fit_feature_expressions(bounds, candidate_features)

    # return bounding_expressions
    return dict()


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
