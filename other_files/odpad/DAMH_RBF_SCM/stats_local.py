
import multiprocessing as mp
import numpy as np
import pandas as pd
import time

import process_worker
import SamplingFramework
import ModelGauss
import MHsampler
import lhs_norm
import scm, rbf
import Stage


## START CONFIGURATION
group_ids = [0,1,2,3,4,5,6] # example: group_ids = [0,1,1,1,2,2,2]
no_chains = 1
no_parameters = 2
no_observations = 1
priorMean = np.ones(no_parameters)*5
priorStd = np.ones(no_parameters)*1.5
noiseStd = np.ones(no_observations)*2*0.0001
proposalStd = np.ones(no_parameters)*1.5
observation = None
artificial_real_parameters = np.ones(no_parameters)*5.1
max_snapshots = 1000 # max. snapshots to construct a surrogate model
min_snapshots = 35 # min. snapshots to construct a surrogate model
s1=Stage.Stage('MH', 30, 10, 'MH_0')
s2=Stage.Stage('DAMHSMU', 30, 1000, 'DAMHSMU_0')
s3=Stage.Stage('MH', 30, 10, 'MH_1')
s4=Stage.Stage('DAMHSMU', 30, 1000, 'DAMHSMU_1')
stages = [s1, s2, s3, s4]
surr_type = 'SCM'
surrogate_parameters = [no_parameters,7,None,None,None]
#surr_type = 'RBF'
#surrogate_parameters = [None,0,0,0,0]
## FINISH CONFIGURATION)

def solver(data_par):
    k1 = data_par[0]
    k2 = data_par[1]
    f = -0.1
    L = 1.0
    M = 0.5
    D1 = (f*L)/k2
    C1 = D1*k2/k1
    D2 = -f/(2*k1)*(M*M)+C1*M+f/(2*k2)*(M*M)-D1*M
    uL = -f/(2*k2)*(L*L)+D1*L+D2
    return uL

def master_local(Model,SF,G_data,no_solvers,group_leader_ids):
    print('Master process communicates with solver(s).')
    free_solvers = [True] * no_solvers
    from_child = [0] * no_solvers
    parameters = [0] * no_solvers
    data_par = np.zeros(Model.no_parameters)
    no_evaluations = 0
#    bound_MH = 100   # HARD CODED !
    while SF.shared_finisher.value < (len(SF.stages)+SF.no_chains):
        time.sleep(0.1)
        # if (full solver) queue is not empty, sends request to free solver (if any)
        if not SF.shared_queue_solver.empty():# == False:
            if True in free_solvers:
                num_child, data_par = SF.shared_queue_solver.get()
                id_solver = free_solvers.index(True)
                from_child[id_solver] = num_child
                parameters[id_solver] = data_par
                data_obs = solver(data_par)
                free_solvers[id_solver] = False
                SF.shared_parents_solver[from_child[id_solver]].send(data_obs)
#                    SF.shared_queue_surrogate.put([parameters[id_solver],data_obs,data_wei])
                G_data = np.vstack([G_data,np.append(data_obs,parameters[id_solver])])
                free_solvers[id_solver] = True
                no_evaluations += 1
        # signal to switch from MH to DAMH-SMU
#        if no_evaluations >= bound_MH:
#            with SF.shared_finisher.get_lock():
#                if SF.shared_finisher.value == 0:
#                    SF.shared_finisher.value = 1
#                    print("FINISHER VALUE CHANGED TO", SF.shared_finisher.value, "(by master)")
    # sends finishing signals to all solvers
    G_DataFrame = pd.DataFrame(G_data)
    G_DataFrame.to_csv('G_data_linela_MH.csv')

no_solvers = max(group_ids)
group_leader_ids = []
for i in range(no_solvers):
    group_leader_ids.append(group_ids.index(i+1))

Model = ModelGauss.ModelGauss(no_parameters, no_observations, priorMean, priorStd, noiseStd, observation)
MHsampler.MHsampler.Model = Model

if Model.observation == None:
    artificial_observation_without_noise = solver(artificial_real_parameters)
    Model.SetNoisyObservation(artificial_observation_without_noise)
    print('artificial_observation_without_noise:',artificial_observation_without_noise)
    print('artificial observation with noise:',Model.observation)

SF = SamplingFramework.SamplingFramework(no_chains, stages, max_snapshots, min_snapshots)
MHsampler.MHsampler.SF = SF
initial_samples = lhs_norm.lhs_norm(Model,SF.no_chains)
G_data = np.zeros([0,no_observations+no_parameters])
if surr_type == 'SCM':
    Surrogate = scm.scm()
else:
    Surrogate = rbf.rbf()
Surrogate.parameters = surrogate_parameters
#    G_data = np.vstack([G_data,np.append(artificial_observation_without_noise,artificial_real_parameters)]) # should not be known
#    SF.shared_queue_surrogate.put([artificial_real_parameters,artificial_observation_without_noise]) # should not be known
if __name__ == '__main__':
    jobs = []
    for i in range(SF.no_chains + SF.no_helpers):
        p = mp.Process(target=process_worker.worker, args=(i,Model,SF,Surrogate,no_solvers,initial_samples,group_leader_ids,proposalStd))
        p.start()
        jobs.append(p)
master_local(Model,SF,G_data,no_solvers,group_leader_ids)
if __name__ == '__main__':
    for i in range(SF.no_chains + SF.no_helpers):
        print("Joining",i)
        jobs[i].join()
## FINISH SAMPLING PROCESS
print('Master finished')
SF.Finalize()


