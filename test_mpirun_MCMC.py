#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:30:07 2020

@author: ber0061
"""

"""mpi4py needs to be compiled with the exact version of Open MPI 
that you are using. E.g., if you have upgraded Open MPI, you need 
to re-compile mpi4py (depending on how you installed mpi4py, 
you may need to uninstall it and re-install it to force it to 
recompile against your new version of Open MPI)."""

import os
syspath = os.path.dirname(os.path.realpath(__file__))

from configuration import Configuration
C = Configuration()

#print('Running code')
#com = 'mpirun -n 3 ' + str(sys.executable) + " " + syspath + "/test_mpi.py"
#com = 'mpirun -n 1 --oversubscribe python3 -m mpi4py process_SAMPLER.py : -n 1 python3 process_SOLVER.py : -n 1 python3 process_COLLECTOR.py'
#com = 'mpirun -n 3 --oversubscribe python3 -m mpi4py process_SAMPLER.py : -n 1 python3 process_COLLECTOR.py'
#com = 'mpirun -n 1 python3 tester_COLLECTOR.py : -n 1 python3 process_COLLECTOR_torso.py'
process = lambda numproc, filename : ' -n ' + str(numproc) + ' --oversubscribe python3 -m mpi4py ' + filename + ' '
com = 'mpirun' + process(C.no_samplers,'process_SAMPLER.py') + ':' + process(1,'process_SOLVER.py') + ':' + process(1,'process_COLLECTOR.py')
print(com)
#os.system(com)
#print('Done')