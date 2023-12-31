from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = random.randint(1, 100)
    print(f"Process {rank}: Valeur generee aleatoirement = {data}", flush=True)
    comm.send(data, dest=(rank + 1) % size)  # Envoie au successeur
else:
    data = comm.recv(source=(rank - 1) % size)  # Réception du prédécesseur
    print(f"Process {rank}: Message recu = {data}", flush=True)
    comm.send(data, dest=(rank + 1) % size)  # Envoie au successeur

# Attendre que le message atteigne à nouveau le processus de rang 0
if rank == 0:
    final_data = comm.recv(source=size - 1)
    print(f"Process {rank}: Message final recu = {final_data}", flush=True)
else:
    comm.send(data, dest=(rank + 1) % size)  # Envoie au successeur
