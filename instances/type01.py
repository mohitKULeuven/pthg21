from cpmpy import *
import json

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

    # arcs diff color
    for pdict in inputData["list"]:
        m += (ivars[pdict['nodeA']] != ivars[pdict['nodeB']])

    m.maximize(max(ivars))

    return (ivars,m)

if __name__ == "__main__":
    i = 1
    fjson = f"type01/instance{i}.json"

    djson = json.load(open(fjson))
    (v, m) = model_type01(**djson)
    print(m)
