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

if visualize:
    ### LOAD CONFIGURATION:
    conf_path = "examples/" + problem_name + ".json"
    with open(conf_path) as f:
        conf = json.load(f)
    if "visualization_filename" in conf.keys(): # if visualization_filename is given
        if "visualization_path" in conf.keys():
            visualization_path = conf["visualization_path"] + conf["visualization_filename"]
        else:
            visualization_path = "examples/visualization/" + conf["visualization_filename"]
    else:
        visualization_path = "examples/visualization/" + problem_name + ".py" # if "visualization/problem_name.py" exists
        if not os.path.isfile(visualization_path): # else use general visualization script
            visualization_path = "examples/visualization/general_visualization.py"
    command = "python3 " + visualization_path + " " + str(N) + " " + problem_name
else:
    if oversubscribe:
        opt = " --oversubscribe "
    else:
        opt = " "
    sampler = " -n " + str(N) + opt + "python3 -m mpi4py surrDAMH/process_SAMPLER.py "
    solver = " -n 1" + opt + "python3 -m mpi4py surrDAMH/process_SOLVER.py " + problem_name + " "
    collector = " -n 1" + opt + "python3 -m mpi4py surrDAMH/process_COLLECTOR.py "
    command = "mpirun" + sampler + ":" + solver + ":" + collector

print(command)
os.system(command)