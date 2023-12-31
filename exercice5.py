from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calculate_pi_part(start, end, step):
    partial_sum = 0.0
    for i in range(start, end):
        x = (i + 0.5) * step
        partial_sum += 4.0 / (1.0 + x**2)
    return partial_sum

def calculate_pi_mpi(nb_iters):
    global_sum = np.zeros(1, dtype=np.float64)  # Initialisation d'un tableau NumPy pour stocker le résultat final
    local_partial_sum = np.zeros(1, dtype=np.float64)  # Initialisation d'un tableau NumPy pour stocker la somme partielle locale

    # Broadcast le nombre d'itérations à tous les processus
    nb_iters = comm.bcast(nb_iters, root=0)

    # Calcule le nombre d'itérations par processus
    local_nb_iters = nb_iters // size

    # Calcule la somme partielle locale en parallèle avec OpenMP
    start = rank * local_nb_iters
    end = (rank + 1) * local_nb_iters
    step = 1.0 / nb_iters

    local_partial_sum[0] = calculate_pi_part(start, end, step)

    # Réduit les sommes partielles locales en la somme globale
    comm.Reduce(local_partial_sum, global_sum, op=MPI.SUM, root=0)

    # Le processus maître (rang 0) calcule et imprime le résultat final
    if rank == 0:
        pi_result = global_sum[0] / nb_iters
        print(f"Approximated value of pi: {pi_result}")

if __name__ == "__main__":
    nb_iters = 1000000  # Ajustez le nombre d'itérations selon vos besoins

    calculate_pi_mpi(nb_iters)
