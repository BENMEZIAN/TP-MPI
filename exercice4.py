from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 1000000
root = 0

if rank == root:
    data = np.random.rand(n)
else:
    data = np.empty(n, dtype=np.float64)

# MPI_Bcast
comm.Barrier()
start_time = time.time()
comm.Bcast(data, root=root)
comm.Barrier()

if rank == 0:
    print(f"MPI_Bcast Time: {time.time() - start_time:.6f} seconds")

"""
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 1000000
root = 0

if rank == root:
    data = np.random.rand(n)
else:
    data = None

# Bloquante
comm.Barrier()
start_time = time.time()

if rank == root:
    for i in range(size):
        if i != rank:
            comm.send(data, dest=i)
else:
    data = comm.recv(source=root)

comm.Barrier()
if rank == 0:
    print(f"Blocking Broadcast Time: {time.time() - start_time:.6f} seconds")
"""