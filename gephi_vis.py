import numpy as np
from imports import Adjac_file_man, P_to_CIJ, adjacency_maker


def unique_matrices():
    N, K, m, h = 512, 3, 4, 3
    E = N*K
    file = open(f"IC/MHN/gephi/nodes_N={N}_K={K}_m={m}_h={h}.csv", 'w')
    file.write("Label\tID\n")
    for i in range(N):
        file.write(str(i)+"\t"+str(i)+"\n")
    file.close()


    P, p_crit, net = Adjac_file_man(N, E, m, h, 'w')
    CIJ = P_to_CIJ(P)
    file2 = open(f"IC/MHN/gephi/edges_N={N}_K={K}_m={m}_h={h}.csv", 'w')
    file2.write("Source\tTarget\tType\tWeight\n")
    for j in range(N):
        for k in range(N):
            if CIJ[j,k] == 1:
                file2.write(str(j)+"\t"+str(k)+"\t"+"Directed\t"+"1\n")

    file2.close()

def weighted_matrices():
    N, K, m, h = 1024, 3, 5, 5
    E = N*K
    file = open(f"IC/MHN/gephi/weighted_nodes_N={N}_K={K}_m={m}_h={h}.csv", 'w')
    file.write("Label\tID\n")
    for i in range(N):
        file.write(str(i)+"\t"+str(i)+"\n")
    file.close()

    CIJ = np.zeros((N,N))
    for l in range(20):
        CIJ += adjacency_maker(N, E, m, h)
    file2 = open(f"IC/MHN/gephi/weighted_edges_N={N}_K={K}_m={m}_h={h}.csv", 'w')
    file2.write("Source\tTarget\tType\tWeight\n")
    for j in range(N):
        for k in range(N):
            if CIJ[j,k] != 0:
                file2.write(str(j)+"\t"+str(k)+"\t"+"Directed\t"+str(CIJ[j,k])+"\n")

    file2.close()

unique_matrices()