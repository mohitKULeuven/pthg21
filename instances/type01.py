from cpmpy import *
import json
import sys
import os
import time
from instance import Instance

"""
    Graph coloring

    Arguments:
    - formatTemplate: dict as in the challenge, containing:
        "list": list of dicts of 'high', 'low', 'type'
    - inputData: dict as in challenge, containing:
        "list": list of dicts with 'nodeA', 'nodeB'
"""
def model_type01(formatTemplate, inputData, **kwargs):

    m = Model()

    # prep vars
    ivars = []
    vname = "list"
    for i,vdict in enumerate(formatTemplate[vname]):
       ivars.append( intvar(vdict['low'], vdict['high'], name=f"{vname}{i}") )  
    ivars = cpm_array(ivars)

    # arcs diff color
    for pdict in inputData["list"]:
        m += (ivars[pdict['nodeA']] != ivars[pdict['nodeB']])

    m.maximize(max(ivars))

    return ({vname:ivars},m)

if __name__ == "__main__":
    i = 1
    json_file = f"type01/instance{i}.json"

    json_data = json.load(open(json_file))
    (v, m) = model_type01(**json_data)
    print("Ground truth model:")
    print(m)

    # check solutions
    if json_data["solutions"]:
        sys.path.insert(1, os.path.realpath(os.path.pardir))
        from learner import check_solutions

        # without objective
        m2 = Model(m.constraints)
        print("Check positives:")
        t0 = time.time()
        # check_solutions and is_sat have strong assumptions
        # on 'sols' and 'v', does not accept dicts?
        # manually...
        cnt = 0
        for dsol in json_data['solutions']:
            vals = dsol['list']
            obj = dsol['objective']
            s = SolverLookup.get("ortools", m2)
            s += (v['list'] == vals)
            cnt += s.solve()
            # OK, not sure how to handle objective well
            # which is not our focus anyway
        print(cnt, "of", len(json_data['solutions']))
        print("\t in",time.time()-t0)

        # Alternative, single call method...
        from cpmpy_helper import solveAll
        print("Alternative method, positives")
        t0 = time.time()
        sols = []
        for dsol in json_data['solutions']:
            sols.append( dsol['list'] )
        s = SolverLookup.get("ortools", m2)
        # add table: only allow solutions in 'sols'
        s += Table(v['list'], sols)
        cnt = solveAll(s)
        print(cnt, "of", len(json_data['solutions']))
        print("\t in",time.time()-t0)

        # Same for negatives, though to make
        # this work one should first filter out those
        # that do not satisfy the objective
        print("Alternative method, negatives")
        t0 = time.time()
        sols = []
        for dsol in json_data['nonSolutions']:
            # XXX: filter out if violate objective
            sol = dsol['list']
            if max(sol) == dsol['objective']:
                sols.append( dsol['list'] )
        s = SolverLookup.get("ortools", m2)
        # add table: only allow solutions in 'sols'
        s += Table(v['list'], sols)
        cnt = solveAll(s)
        print(cnt, "of", len(json_data['nonSolutions']))
        print("\t in",time.time()-t0)
        

