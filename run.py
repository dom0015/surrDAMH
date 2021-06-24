# -*- coding: utf-8 -*-

import os
import sys
import json

"""
python3 run.py problem_name N (oversubscribe) (visualize)
example: python3 run.py simple 4 oversubscribe
"""

### DEFAULT PARAMETERS:
problem_name = "simple" # requires configuration file "conf/" + problem_name + ".json"
# prepared toy examples: "simple", "simple_MPI", "Darcy"
N = 4 # number of sampling processes
oversubscribe = False # if there are not enough slots available
visualize = False # True = only visualization

### PARSE COMMAND LINE ARGUMENTS: 
len_argv = len(sys.argv)
if len_argv>1:
    problem_name = sys.argv[1]
if len_argv>2:
        N = int(sys.argv[2]) # number of MH/DAMH chains
if len_argv>3:
    if sys.argv[3] == "oversubscribe":
        oversubscribe = True
    else:
        oversubscribe = False
    if sys.argv[3] == "visualize":
        visualize = True
    else:
        visualize = False

problem_path = None
basename = os.path.basename(problem_name)
fname, fext = os.path.splitext(basename)
if fext == ".json":
    problem_path = os.path.abspath(problem_name)
    problem_name = fname
elif fext == "":
    problem_path = os.path.abspath(os.path.join("examples", problem_name + ".json"))
else:
    os.error("Specify configuration json file or example testcase name.")

with open(problem_path) as f:
    conf = json.load(f)

if visualize:
    if os.path.exists("examples/visualization/" + problem_name + ".py"):
        command = "python3 examples/visualization/" + problem_name + ".py " + str(N)
    else:
        command = "python3 examples/visualization/general_visualization.py " + str(N) + " " + problem_name
else:
    if oversubscribe:
        opt = " --oversubscribe " 
    else:
        opt = " "
    # opt = opt + "--mca opal_warn_on_missing_libcuda 0 "
    # opt = opt + "--mca orte_base_help_aggregate 0 "
    sampler = " -n " + str(N) + opt + "python3 -m mpi4py surrDAMH/process_SAMPLER.py "
    solver = " -n 1" + opt + "python3 -m mpi4py surrDAMH/process_SOLVER.py " + problem_path + " "
    collector = " -n 1" + opt + "python3 -m mpi4py surrDAMH/process_COLLECTOR.py "
    if "surrogate_type" in conf.keys():
        command = "mpirun" + sampler + ":" + solver + ":" + collector
    else:
        command = "mpirun" + sampler + ":" + solver

# path = os.path.abspath(os.path.dirname(__file__)) # file directory 
# sys.path.append(path)
# print("path append:", path)
# print(sys.path)
print(command)
os.system(command)

# comment