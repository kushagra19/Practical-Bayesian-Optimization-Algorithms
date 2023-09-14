import os
import time

import yaml
import torch
# from mpi4py import MPI

from BO import BO
from test_functions import *
from CustomAcquistionFunction import *

class Experiment():
    def __init__(self, is_major_change, config, dtype, mode='train'):
        self.is_major_change = is_major_change
        self.config = config
        self.obj_fn = self._get_appropriate_func()
        self.dtype = dtype

        self.init_examples = config["init_examples"]
        base_size = self.config["budget"] + self.init_examples
        self.X = torch.empty((
            base_size,
            self.config["dims"],
            self.config["runs"]
        ))
        self.y = torch.empty((
            base_size,
            1,
            self.config["runs"]
        ))
        self.grads = torch.empty((
            base_size,
            self.config["dims"],
            self.config["runs"]
        ))

        self.mode = mode

        if self.mode == 'train':
            self._init_directory_structure()


    def _init_directory_structure(self):
        '''
            TO-DO: If major change then copy the code as well.
        '''
        try:
            self.exp_dir = os.path.join(self.config["exp_dir"],
                self.config["objective_function"],
                self.config["experiment_name"]
            )
            os.makedirs(self.exp_dir)
            # shutil.copy("./config.json", self.exp_dir)

            summary = {}
            summary.update(self.config)

            config_file = os.path.join(self.exp_dir, "config.yaml")
            with open(config_file, "w") as outfile:
                yaml.dump(summary, outfile)
        except Exception as e:
            import traceback
            traceback.print_exc()


    def _get_appropriate_func(self):
        if self.config["objective_function"] == 'le_branke':
            return LeBranke(
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        if self.config["objective_function"] == 'branin':
            return Branin(
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'levy':
            return Levy(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'rosenbrock':
            return Rosenbrock(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'ackley':
            return Ackley(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'hartmann':
            return Hartmann(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'powell':
            return Powell(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'holder_table':
            return Holder_Table(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'gw':
            return gw(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'Cosine8':
            return Cosine8(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'Drop_Wave':
            return Drop_Wave(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'DixonPrice':
            return DixonPrice(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        
        if self.config["objective_function"] == 'StyblinskiTang':
            return StyblinskiTang(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'SixHumpCamel':
            return SixHumpCamel(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'ThreeHumpCamel':
            return ThreeHumpCamel(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'Sphere':
            return Sphere(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        
        if self.config["objective_function"] == 'Bukin':
            return Bukin(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        
        if self.config["objective_function"] == 'Griewank':
            return Griewank(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        if self.config["objective_function"] == 'Michalewicz':
            return Michalewicz(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        if self.config["objective_function"] == 'Sum_exp':
            return Sum_exp(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'cart_pole':
            return cart_pole(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        
        if self.config["objective_function"] == "illustrationNd":
            return IllustrationND(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"]
            )

        if self.config["objective_function"] == 'pongChangeMountainCar':
            return pongChangeMountainCar(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'lunar_lander':
            return lunar_lander(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == "rotation_transformer":
            return RotationTransformation()    
        

    def perform_experiment(self, rank):
        grad_acq = (PrabuAcquistionFunction if
            self.config["grad_acq_func"] == "PrabuAcquistionFunction" else
            SumGradientAcquisitionFunction)

        torch.manual_seed(rank+1)

        bo = BO(
            obj_fn=self.obj_fn,
            dtype=self.dtype,
            acq_func=self.config["zobo_acq_func"],
            grad_acq=grad_acq,
            init_examples=self.init_examples,
            order=self.config["order"],
            budget=self.config["budget"],
            query_point_selection=self.config["query_point_selection"],
            num_restarts=self.config["num_restarts"],
            raw_samples=self.config["raw_samples"],
            grad_acq_name=self.config["grad_acq_func"]
        )

        X, y,y_true, grads = bo.optimize()

        if self.config["max/min"] == "min":
            y = -y

        self.save_results(rank, X, y.unsqueeze(dim=-1),y_true.unsqueeze(dim=-1), grads)


    def save_results(self, rank, X, y,y_true, grads=None):
        torch.save(X, os.path.join(self.exp_dir,  f'/X_{rank}.pt'))
        torch.save(y, os.path.join(self.exp_dir , f'/y_{rank}.pt'))
        torch.save(y_true, os.path.join(self.exp_dir , f'/y_true_{rank}.pt'))
        if grads is not None:
            torch.save(grads, os.path.join(self.exp_dir, f'/grads_{rank}.pt'))

    def sched_jobs(self,comm,free_ranks,required,l):
        used = []
        minimum = min(len(free_ranks),len(required))
        for i in range(minimum):
            job_details = {'job_id':i+l}
            comm.send(job_details,dest =free_ranks[i])
            used.append(free_ranks[i])
        tmp = []
        for i in free_ranks:
            if i not in used:
                tmp.append(i)
        completed = minimum
        free_ranks = tmp
        return free_ranks,completed

    def multi(self):
        comm = MPI.COMM_WORLD
        n_processors = comm.Get_size()
        rank = comm.Get_rank()
        name = MPI.Get_processor_name()
        n_runs = self.config["runs"]

        if rank == 0:
            free_ranks = [i for i in range(1,n_processors)]
            required = [i for i in range(n_runs)]

            if(len(required) <= len(free_ranks)):
                free_ranks,completed = self.sched_jobs(
                    comm,
                    free_ranks,
                    required,
                    0
                )
                time.sleep(2)
                print("Total Jobs Added...",completed)
            else:
                free_ranks,completed = self.sched_jobs(
                    MPI.COMM_WORLD,
                    free_ranks,
                    required[0:len(free_ranks)],
                    0
                )
                total_completed = completed
                print("Total Jobs Added...",total_completed)
                done = False
                while not done:
                    time.sleep(2)
                    for pr in range(1, n_processors):
                        if comm.iprobe(pr):
                            comm.recv(source=pr)
                            free_ranks.append(pr)
                    tmp = total_completed
                    free_ranks,completed = self.sched_jobs(
                        MPI.COMM_WORLD,
                        free_ranks,
                        required[total_completed:],
                        total_completed
                    )
                    total_completed += completed
                    if tmp != total_completed:
                        print("Total Jobs Added...",total_completed)
                    if total_completed ==len(required):
                        done = True
            time.sleep(10)

            killed_ranks = []

            while len(killed_ranks) < n_processors - 1:
                for krank in free_ranks:
                    task_kill = {'exit': True}
                    comm.send(task_kill, krank)
                    killed_ranks.append(krank)
                free_ranks = []
                time.sleep(2)
                for pr in range(1, n_processors):
                    if pr not in killed_ranks and comm.iprobe(pr):
                        comm.recv(source=pr)
                        free_ranks.append(pr)
            print("Ranks Killed...",len(killed_ranks))
            print("Rank 0 exiting...")

        else:
            name = MPI.Get_processor_name()
            while True:
                job_details = comm.recv(source=0)
                if 'exit' in job_details:
                    break

                self.perform_experiment(job_details['job_id'])

                comm.send(0, dest=0)
