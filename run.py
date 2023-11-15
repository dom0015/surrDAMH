# -*- coding: utf-8 -*-

import os
import sys
import json

"""
Usage:
    python3 run.py problem_path N output_dir (oversubscribe) (visualize)
Args:
    problem_path (str): path to yaml/json configuration file
    N (int): number of SAMPLERs (MPI processes running MH algorithms)
    output_dir: path to folder for storing output data (samples etc.)
Example:
    python3 run.py examples/simple.json 4 test_dir oversubscribe
"""

# DEFAULT PARAMETERS:
problem_path = "examples/simple.json"
# prepared toy examples: "simple", "simple_MPI", "Darcy"
N = 2
output_dir = "test_dir"
oversubscribe = False  # True if there are not enough slots available (in Open MPI)
visualize = False  # True = only visualization (TODO)

# PARSE COMMAND LINE ARGUMENTS:
len_argv = len(sys.argv)
if len_argv > 1:
    problem_path = sys.argv[1]
if len_argv > 2:
    N = int(sys.argv[2])
if len_argv > 3:
    output_dir = sys.argv[3]
if len_argv > 4:
    if sys.argv[4] == "oversubscribe":
        oversubscribe = True
    else:
        oversubscribe = False
    if sys.argv[4] == "visualize":
        visualize = True
    else:
        visualize = False

#problem_path = None
#basename = os.path.basename(problem_name)
#fname, fext = os.path.splitext(basename)
#if fext == ".json":
#    problem_path = os.path.abspath(problem_name)
#    problem_name = fname
#elif fext == "":
#    problem_path = os.path.abspath(os.path.join("examples", problem_name + ".json"))
#else:
#    os.error("Specify configuration json file or example testcase name.")

with open(problem_path) as f:
    conf = json.load(f)

if visualize:  # TODO
    args = [str(N), problem_path, output_dir]
    file_path = os.path.dirname(os.path.abspath(__file__))
    problem_basename = os.path.basename(problem_path)
    visualization_path = file_path + \
        "/examples/visualization/" + problem_basename + ".py"
    if os.path.exists(visualization_path):
        command = "python3 " + visualization_path + " " + " ".join(args)
    else:
        visualization_path = os.path.abspath(os.path.join(
            "examples/visualization/general_visualization.py"))
        command = "python3 " + visualization_path + " " + " ".join(args)
else:
    if oversubscribe:
        opt = " --oversubscribe "
    else:
        opt = " "
    # opt = opt + "--mca opal_warn_on_missing_libcuda 0 "
    # opt = opt + "--mca orte_base_help_aggregate 0 "
    sampler = " -n " + \
        str(N) + opt + "python3 -m mpi4py surrDAMH/process_SAMPLER.py " + \
        output_dir + " "
    solver = " -n 1" + opt + "python3 -m mpi4py surrDAMH/process_SOLVER.py " + \
        problem_path + " " + output_dir + " "
    collector = " -n 1" + opt + "python3 -m mpi4py surrDAMH/process_COLLECTOR.py "
    if "surrogate_type" in conf.keys():
        command = "mpirun" + sampler + ":" + solver + ":" + collector
    else:
        command = "mpirun" + sampler + ":" + solver

# path = os.path.abspath(os.path.dirname(__file__)) # file directory
# sys.path.append(path)
# print("path append:", path)
# print(sys.path)

import sys
# sys.path.append("/home/domesova/GIT/Endorse-2Dtest-Bayes/surrDAMH")
print(command)
os.system(command)

# comment
