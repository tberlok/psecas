class IO:
    def __init__(self, solver, data_folder, experiment, steps, tag=''):
        """
        Initialisation creates the output directory and saves the path to the
        io object.
        It also copies the experiment script to the data directory and
        saves a dictionary with important information.
        """
        from mpi4py.MPI import COMM_WORLD as comm
        from mpi4py.MPI import Wtime
        from numpy import arange

        self.solver = solver

        self.steps = steps
        self.index_global = arange(steps)
        self.index_local = self.index_global[comm.rank::comm.size]
        self.steps_local = len(self.index_local)

        if comm.rank == 0:
            import subprocess
            from datetime import datetime
            from numpy import float64

            # Create datafolder
            if data_folder[-1] != '/':
                data_folder += '/'
            subprocess.call('mkdir ' + data_folder, shell=True)

            # Copy the experiment to the data folder
            # experiment = path.basename(experiment)
            subprocess.call('cp ' + experiment + ' ' + data_folder +
                            experiment, shell=True)
            info = {'experiment': experiment}

            # Save git commit number
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            git_commit = git_commit.strip().decode('utf-8')
            info.update({'git_commit': git_commit})

            # Save the hostname
            hostname = subprocess.check_output(['hostname'])
            hostname = hostname.strip().decode('utf-8')
            info.update({'hostname': hostname})

            # Save the time at the start of simulation
            i = datetime.now()
            simulation_start = i.strftime('%d/%m/%Y at %H:%M:%S')
            info.update({'simulation_start': simulation_start})

            # Add tag which can be used to group simulations together
            info.update({'tag': tag})

            # Collect all variables in local namespace that are int, float or
            # float64. These will in general be the values set by us.
            # for key in local_vars.keys():
            #     if type(local_vars[key]) in (float, float64, int):
            #         info.update({key: local_vars[key]})

            # Save the number of MPI processes used
            info.update({'MPI': comm.size})

            # Save the dictionary info to the file info.p
            # pickle.dump(info, open(data_folder+'info.p', 'wb'))

            # Start a log file
            f = open('evp.log', 'w')

            f.write('Log file for EVP run\n')
            f.write('Solving {} evp problems on {} processors\n\n'.format(
                    self.steps, comm.size))
            # Write start date and time
            f.write('Calculation started on ' + simulation_start + '\n\n')
            # Write the contents of info.p for convenience
            f.write('Contents of info.p is printed below \n')
            for key in info.keys():
                f.write(key + ' = {} \n'.format(info[key]))

            f.write('\nContents of system is printed below \n')
            for key in solver.system.__dict__.keys():
                if type(solver.system.__dict__[key]) in (float, int, list):
                    f.write(key+' = {}\n'.format(solver.system.__dict__[key]))

            f.write('\nUsing {} with\n'.format(type(solver.grid)))
            for key in solver.grid.__dict__.keys():
                if type(solver.grid.__dict__[key]) in (float, float64, int):
                    f.write(key+' = {} \n'.format(solver.grid.__dict__[key]))

            f.write('\n\nEntering main calculation loop \n')
            f.close()

        # Folder where data is stored
        self.data_folder = data_folder

        # Used for computing total runtime
        self.wt = Wtime()

    def log(self, i, time, custum_str):
        from mpi4py.MPI import COMM_WORLD as comm
        f = open('evp.log', 'a')
        msg = "Solved EVP with" + custum_str + "in {:1.2f} seconds. \
               Rank {} is {:2.0f}% done.\n"
        f.write(msg.format(time, comm.rank, (i+1)/self.steps_local*100))
        f.close()

    def save_system(self, i):
        import pickle
        file = self.data_folder + 'globalid-{}.p'.format(self.index_local[i])
        self.solver.system.result = self.solver.result
        pickle.dump(self.solver.system, open(file, 'wb'))
        # Delete d0, d1 and d2 for storage effieciency
        system = pickle.load(open(file, 'rb'))
        del system.grid.d0
        del system.grid.d1
        del system.grid.d2
        pickle.dump(system, open(file, 'wb'))

    def finished(self):
        """Write elapsed time to log file and move the log file to the data
        directory"""
        from mpi4py.MPI import COMM_WORLD as comm
        from mpi4py.MPI import Wtime
        seconds = Wtime() - self.wt
        if comm.rank == 0:
            import subprocess
            from datetime import datetime

            # Time at end of simulation
            i = datetime.now()
            endtime = i.strftime('%d/%m/%Y at %H:%M:%S')

            f = open('evp.log', 'a')
            f.write('Calculation ended on '+endtime+'\n')
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)

            msg = 'Time elapsed was {} days {} hours {} minutes {:1.4} seconds'
            f.write(msg.format(d, h, m, s))
            f.close()

            subprocess.call('mv evp.log ' + self.data_folder, shell=True)
