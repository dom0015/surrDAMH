
from mpi4py import MPI

import multiprocessing as mp
import numpy as np

import process_master
import process_worker
import SamplingFramework
import ModelGauss
import MHsampler
import lhs_norm
import scm, rbf
import Stage

## START CONFIGURATION
seed = 6
group_ids = [0,1,2,3,4,5,6] # example: group_ids = [0,1,1,1,2,2,2]
no_chains = 1
no_parameters = 2
no_observations = 1
priorMean = np.ones(no_parameters)*10
priorStd = np.ones(no_parameters)*1.5
noiseStd = np.ones(no_observations)*2*0.0001
proposalStd = np.ones(no_parameters)*0.5
observation = None
artificial_real_parameters = np.ones(no_parameters)*8
max_snapshots = 5000 # max. snapshots to construct a surrogate model
min_snapshots = 5 # min. snapshots to construct a surrogate model
s1=Stage.Stage('MH', 10, 5000, 'MH_0')
s2=Stage.Stage('DAMHSMU', 10, 5000, 'DAMHSMU_0')
#s3=Stage.Stage('ADAMH', 20, 1000, 'ADAMH_0')
#s4=Stage.Stage('DAMHSMU', 10, 1000, 'DAMHSMU_1')
stages = [s1, s2]
#surr_type = 'SCM'
#surrogate_parameters = [no_parameters,7,None,None,None]
surr_type = 'RBF'
surrogate_parameters = [None,0,0,0,0]
## FINISH CONFIGURATION

comm_world = MPI.COMM_WORLD
group_world = comm_world.Get_group()
rank_world = comm_world.Get_rank()

no_solvers = max(group_ids)
group_leader_ids = []
for i in range(no_solvers):
    group_leader_ids.append(group_ids.index(i+1))
    
comm_world.Barrier()

if rank_world == 0:
    group_local = MPI.Group.Incl(group_world,[])
    comm_local = comm_world.Create(group_local)
else:
    id_group = group_ids[rank_world];
    group_local_ids = [j for j, e in enumerate(group_ids) if e == id_group]
    group_local = MPI.Group.Incl(group_world,group_local_ids)
    comm_local = comm_world.Create(group_local)
    print('Rank_world:',rank_world,'id_group:',id_group,'local rank:',group_local.Get_rank(),'local size:',group_local.Get_size(),'group_local_ids:',group_local_ids,)

comm_world.Barrier()

if rank_world == 0:
    Model = ModelGauss.ModelGauss(no_parameters, no_observations, priorMean, priorStd, noiseStd, observation)
    MHsampler.MHsampler.Model = Model
    
    if Model.observation == None:
        comm_world.Send(np.append(0,artificial_real_parameters), dest=group_leader_ids[0], tag=1)
        received_data = np.zeros(no_observations+no_parameters+1)
        comm_world.Recv(received_data, source=group_leader_ids[0],tag=1)
        Model.SetNoisyObservation(received_data[no_parameters+1:])
        print('artificial_observation_without_noise:',received_data[1:])
        print('artificial observation with noise:',Model.observation)
    
    SF = SamplingFramework.SamplingFramework(no_chains, stages, max_snapshots, min_snapshots, no_solvers)
    MHsampler.MHsampler.SF = SF
    MHsampler.MHsampler.seed = seed
    initial_samples = lhs_norm.lhs_norm(Model,SF.no_chains,seed)
    print("initial samples:",initial_samples)
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
            p = mp.Process(target=process_worker.worker, args=(i,comm_world,Model,SF,Surrogate,no_solvers,initial_samples,group_leader_ids,proposalStd))
            p.start()
            jobs.append(p)
    process_master.master(comm_world,Model,SF,G_data,no_solvers,group_leader_ids)
    if __name__ == '__main__':
        for i in range(SF.no_chains + SF.no_helpers):
            print(jobs[i])
            jobs[i].join()
            print("Join job",i)
## FINISH SAMPLING PROCESS
    print('Master finished')
    SF.Finalize()

if rank_world in group_leader_ids:
    print("Hello, I am group leader with local rank", comm_local.Get_rank(), "and global rank", rank_world )
    finish = 0
    status = MPI.Status()
    received_data = np.zeros(no_parameters + 1)
    sent_data = np.zeros(no_parameters + no_observations + 1)#+rank_world
    while finish == 0:
        comm_world.Recv(received_data, source=0, tag=MPI.ANY_TAG, status=status)
#        print("Solver",rank_world,"received data",received_data)
        num_child = received_data[0]
        data_par = received_data[1:]
        tag = status.Get_tag()
        if tag == 150:
            print("BYE BYE from rank_world", rank_world)
            finish = 1
        else:
            
            k1 = data_par[0]
            k2 = data_par[1]
            f = -0.1
            L = 1.0
            M = 0.5
            D1 = (f*L)/k2
            C1 = D1*k2/k1
            D2 = -f/(2*k1)*(M*M)+C1*M+f/(2*k2)*(M*M)-D1*M
            uL = -f/(2*k2)*(L*L)+D1*L+D2
            sent_data[0] = num_child
            sent_data[1:3] = data_par
            sent_data[3] = uL
            

#            k1 = data_par[0]
#            k2 = data_par[1]
#            k3 = data_par[2]
#            k4 = data_par[3]
#            f = -0.1
#            L = 1.0
#            M12 = 0.25
#            M23 = 0.5
#            M34 = 0.75
#            C4 = (f*L)/k4
#            C3 = C4*k4/k3
#            C2 = C3*k3/k2
#            C1 = C2*k2/k1
#            D1 = 0
#            D2 = -f/k1*M12*M12/2 + C1*M12 + D1 + f/k2*M12*M12/2 - C2*M12
#            D3 = -f/k2*M23*M23/2 + C2*M23 + D2 + f/k3*M23*M23/2 - C3*M23
#            D4 = -f/k3*M34*M34/2 + C3*M34 + D3 + f/k4*M34*M34/2 - C4*M34
#            uL = -f/k4*L*L/2 + C4*L + D4
#            sent_data[0] = num_child
#            sent_data[1:5] = data_par
#            sent_data[5] = uL
            
            comm_world.Send(sent_data, dest=0, tag=1)
#            print("Solver",rank_world, "Send data",sent_data)

comm_world.Barrier()
print("MPI process", rank_world, "finished.")
