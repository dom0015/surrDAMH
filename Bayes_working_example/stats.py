# Make sure to call this first to initialize MPI
from mpi4py import MPI
# from pympi import *
import multiprocessing as mp
import numpy as np
import process_master
import process_worker
#import test_solver
#import time
#import cProfile

import ModelGauss
import MHsampler
import lhs_norm

#def obalka(num,no_observations,no_parameters,no_solvers, no_chains, no_helpers, shared_finisher, shared_queue_solver, shared_children_solver,shared_queue_updater,shared_queue_surrogate_solver,shared_queue_surrogate,shared_parents_surrogate,shared_children_surrogate,group_leader_ids,proposalStd):
#    cProfile.runctx('process_worker.worker(num,no_observations,no_parameters,no_solvers, no_chains, no_helpers, shared_finisher, shared_queue_solver, shared_children_solver,shared_queue_updater,shared_queue_surrogate_solver,shared_queue_surrogate,shared_parents_surrogate,shared_children_surrogate,group_leader_ids,proposalStd)', globals(), locals(), 'prof%d.prof' %num)
    
comm_world = MPI.COMM_WORLD
group_world = comm_world.Get_group()
rank_world = comm_world.Get_rank()

group_ids = [0,1,2,3,4,5,6]
group_leader_ids = []
no_solvers = max(group_ids)

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

no_chains = 1
no_helpers = 3

## START GLOBAL MODEL SETTINGS
no_parameters = 2
no_observations = 1
## FINISH GLOBAL MODEL SETTINGS

comm_world.Barrier()

if rank_world == 0:

## START MODEL SETTINGS
    MHsampler.MHsampler.no_observations = no_observations
    MHsampler.MHsampler.no_parameters = no_parameters
    priorMean = np.ones(no_parameters)*5 # edited ones/zeros *5
    priorStd = np.ones(no_parameters)*1.5
    noiseStd = np.ones(no_observations)*2*0.0001 # edited *0.0001
    proposalStd = np.ones(no_parameters)*1.5
    initial_samples = lhs_norm.lhs_norm(no_parameters,no_chains,priorMean,priorStd)
    observation = -0.01 # edited
    Model = ModelGauss.ModelGauss(no_parameters, no_observations, priorMean, priorStd, noiseStd, observation)
    MHsampler.MHsampler.Model = Model
    
    artificial_real_parameters = np.ones(no_parameters)*1.5
    artificial_observation_without_noise = np.zeros(no_observations)
    comm_world.Send(artificial_real_parameters, dest=group_leader_ids[0], tag=1)
    comm_world.Recv(artificial_observation_without_noise, source=group_leader_ids[0],tag=1)
    G_data = np.zeros([0,no_observations+no_parameters])
    G_data = np.vstack([G_data,np.append(artificial_observation_without_noise,artificial_real_parameters)])
#    test_solver.solve(aftificial_real_parameters,artificial_observation_without_noise)
    print('artificial_observation_without_noise:',artificial_observation_without_noise)
#    Model.SetNoisyObservation(artificial_observation_without_noise)
    Model.observation = -0.01 # edited
    print('artificial observation with noise:',Model.observation)
## FINISH MODEL SETTINGS
## START SAMPLING PROCESS
    shared_finisher = mp.Value('i',0)
    shared_queue_solver = mp.Queue()
    shared_queue_surrogate = mp.Queue()
    shared_queue_updater = mp.Queue()
    shared_queue_surrogate_solver = mp.Queue()
    shared_queue_surrogate.put([artificial_real_parameters,artificial_observation_without_noise])
    shared_parents_solver = []
    shared_children_solver = []
    shared_parents_surrogate = []
    shared_children_surrogate = []
    for i in range(no_chains):
        new_parent, new_child = mp.Pipe()
        shared_parents_solver.append(new_parent)
        shared_children_solver.append(new_child)
        new_parent, new_child = mp.Pipe()
        shared_parents_surrogate.append(new_parent)
        shared_children_surrogate.append(new_child)
    if __name__ == '__main__':
        jobs = []
        for i in range(no_chains + no_helpers):
            p = mp.Process(target=process_worker.worker, args=(i,no_observations,no_parameters,no_solvers,no_chains,no_helpers,initial_samples,shared_finisher,shared_queue_solver,shared_children_solver,shared_queue_updater,shared_queue_surrogate_solver,shared_queue_surrogate,shared_parents_surrogate,shared_children_surrogate,group_leader_ids,proposalStd))
            p.start()
            jobs.append(p)
    process_master.master(comm_world,no_observations,no_parameters,G_data,shared_finisher,no_solvers,no_chains,shared_queue_solver,shared_parents_solver,shared_queue_surrogate,group_leader_ids)
    if __name__ == '__main__':
        for i in range(no_chains + no_helpers):
            print("Joining",i)
            jobs[i].join()
## FINISH SAMPLING PROCESS
    print('Master finished')
    shared_queue_solver.cancel_join_thread()
    shared_queue_surrogate.cancel_join_thread()
    shared_queue_updater.cancel_join_thread()
    shared_queue_surrogate_solver.cancel_join_thread()
    print("Queues empty?", shared_queue_solver.empty(), shared_queue_surrogate.empty(), shared_queue_updater.empty(), shared_queue_surrogate_solver.empty())

if rank_world in group_leader_ids:
    print("Hello, I am group leader with local rank", comm_local.Get_rank(), "and global rank", rank_world )
    finish = 0
    status = MPI.Status()
    data_par = np.zeros(no_parameters)
    data_obs = np.zeros(no_observations)+rank_world
    while finish == 0:
        comm_world.Recv(data_par, source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == 150:
            print("BYE BYE from rank_world", rank_world)
            finish = 1
        else:
#            x = data_par[0]
#            y = data_par[1]
#            data_obs[0]=pow(pow(x,2)+y-11,2)+pow(x+pow(y,2)-7,2)
            
#            a = data_par[0]
#            b = data_par[1]
#            c = data_par[2]
#            d = data_par[3]
#            e = data_par[4]
#            f = data_par[4]
#            g = data_par[4]
#            h = data_par[4]
#            i = data_par[4]
#            j = data_par[4]
#            data_obs[0]=pow(pow(a,2)+b-1,2)+pow(b+pow(c,2)-1,2)+pow(c+pow(a,2)-1,2)
#            data_obs[0]=pow(pow(a,2)+b-1,2)+pow(b+pow(c,2)-1,2)+pow(c+pow(d,2)-1,2)+pow(d+pow(a,2)-1,2)
#            data_obs[0]=pow(pow(a,2)+b-11,2)+pow(b+pow(c,2)-10,2)+pow(c+pow(d,2)-9,2)+pow(d+pow(e,2)-8,2)+pow(e+pow(a,2)-7,2)
#            data_obs[0]=pow(pow(a,2)+b-11,2)+pow(b+pow(c,2)-10,2)+pow(c+pow(d,2)-9,2)+pow(d+pow(e,2)-8,2)+pow(e+pow(f,2)-7,2)+pow(f+pow(g,2)-7,2)+pow(g+pow(h,2)-7,2)+pow(h+pow(i,2)-7,2)+pow(i+pow(j,2)-7,2)+pow(j+pow(a,2)-7,2)
            
            k1 = data_par[0]
            k2 = data_par[1]
            f = -0.1
            L = 1.0
            M = 0.5
            D1 = (f*L)/k2
            C1 = D1*k2/k1
            D2 = -f/(2*k1)*(M*M)+C1*M+f/(2*k2)*(M*M)-D1*M
            uL = -f/(2*k2)*(L*L)+D1*L+D2
            data_obs[0] = uL
            
#            print(k1,k2,data_obs[0])
            
            comm_world.Send(data_obs, dest=0, tag=1)

comm_world.Barrier()
print("Konec",rank_world)
