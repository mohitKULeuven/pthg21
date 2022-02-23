from cpmpy import *
# from cpmpy.solvers import CPM_ortools
from cpmpy.transformations.get_variables import *
from instance import Instance
import numpy as np
from learner import solutions

def nurse_rostering_instance(nurses=5, days=7):
    schedule_instance = Instance(nurses, {"inputData":{}, "formatTemplate":{}, "solutions": None, "tests":{}}, 21)
    schedule_instance.input_data = {'nurses': nurses, 'days': days}
    schedule_instance.constants = {'nurses': nurses, 'days': days}
    schedule_instance.tensors_dim = {'array': (nurses, days)}
    schedule_instance.var_lbs = {'array': np.zeros([nurses, days]).astype(int)}
    schedule_instance.var_ubs = {'array': np.ones([nurses, days]).astype(int)}
    schedule_instance.var_bounds = {
        k: list(zip(schedule_instance.var_lbs[k].flatten(), schedule_instance.var_ubs[k].flatten()))
        for k in schedule_instance.tensors_dim
    }
    # print(schedule_instance.var_lbs, schedule_instance.var_ubs, schedule_instance.tensors_dim)
    m = nurse_rostering_model(schedule_instance)
    schedule_instance.pos_data = []
    for solution in solutions(m, 100):
        solution = np.reshape(solution, (nurses, days))
        schedule_instance.pos_data.append({'array': solution})
    schedule_instance.training_data = {
        k: np.array([d[k] for d in schedule_instance.pos_data])
        for k in schedule_instance.tensors_dim
    }
    return schedule_instance

def nurse_rostering_model(instance:Instance):
    nurses=instance.input_data['nurses']
    days=instance.input_data['days']
    schedule = instance.cp_vars["array"]
    m = Model()
    m += [sum(schedule[i, :]) >= days-4 for i in range(len(schedule))]
    m += [sum(schedule[i, :]) <= days-1 for i in range(len(schedule))]
    m += [sum(schedule[:, i]) >= nurses-3 for i in range(len(schedule[0]))]
    m += [sum(schedule[:, i]) <= nurses-1 for i in range(len(schedule[0]))]

    return m


if __name__ == "__main__":
    train_instance = nurse_rostering_instance(2, 5)
    m = nurse_rostering_model(train_instance)
    cp_vars = get_variables_model(m)
    s = SolverLookup.get("ortools", m)
    while s.solve():
        print(cp_vars.value())
        s += ~all([var == var.value() for var in cp_vars])
