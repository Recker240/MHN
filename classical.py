import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from tqdm import tqdm
from numba import jit
import networkx as nx

from imports import *
qtde_param = 12

@jit(nopython=True)
def mediador_F_auto(P, p, systems, T, wni, qni):
    F_list = np.zeros(systems)
    rho_std, F_std = 0, 0
    for j in range(systems):
        x_fin, rho = P_time_evaluation(P, p, states, T, wni, qni)
        rho_not_trans = rho[int(0.3*len(rho)):]
        F_list[j] = mean(rho_not_trans)/T
    F = np.mean(F_list)
    F_std = np.std(F_list)
    return F, F_std

def node_index(CIJ, states, T, wni, qni):
    N = len(CIJ)
    fig, ax = plt.subplots()
    at_excit_count = 0
    
    for i in range(T):
        if i == 0:
            x = initial_position(N, wni, qni)
            ot_excit_count = qni
        else:
            x, ot_excit_count = iterator_count(CIJ, x, states)
        for j in range(N):
            ax.plot(j,i,'b.', ms=0.6) if x[j] == 1 else ...
    at_excit_count += ot_excit_count
    
    ax.set_title("CIJ Iterator")
    ax.set_ylabel("T")
    ax.set_xlabel("Nó")
    ax.set_ylim((-0.01*T,1.01*T))
    ax.set_xlim((-0.01*N,1.01*N))
    return x

def P_node_index(P, p, states, T, wni, qni):
    N, k = P.shape
    fig, ax = plt.subplots()
    at_excit_count = 0
    
    for i in range(T):
        if i == 0:
            x = initial_position(N, wni, qni)
            ot_excit_count = qni
        else:
            x, ot_excit_count = P_iterator_count(P, x, states, p)
        for j in range(N):
            ax.plot(j,i,'b.', ms=0.6) if x[j] == 1 else ...
        at_excit_count += ot_excit_count
        
    ax.set_title(f"Iteration for p={p}")
    ax.set_ylabel("T")
    ax.set_xlabel("Nó")
    ax.set_ylim((-0.01*T,1.01*T))
    ax.set_xlim((-0.01*N,1.01*N))
    return x

def file_opener(N,E,m,h,p0,pf,len_p,net,sys_per_p,T):
    folder_adress = f"MHN/Data/mean_rho/N={N}"
    
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"MHN/Data/mean_rho"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"MHN/Data/mean_rho/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"MHN/Data/mean_rho/N={N}/K={E//N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    datas = os.listdir(folder_adress)
    adresses = []
    for arq in datas:
        qtos_ = 0
        for i, carac in enumerate(arq):
            if carac=='_':
                qtos_ += 1
            if qtos_ == 3:
                pos_ = i
                break
        if arq[:pos_] == f"n={net}_T={T}_sysperp={sys_per_p}":
            adresses.append(folder_adress + "/" + arq)
            break
    
    if len(adresses) == 0:
        file = open(folder_adress+f"/n={net}_T={T}_sysperp={sys_per_p}_p=({p0},{pf},0).txt", 'w')
        file.write("ps_generated = 0   \n")
        file.write("N = "+str(int(N))+"\n")
        file.write("E = "+str(int(E))+"\n")
        file.write("m = "+str(int(m))+"\n")
        file.write("h = "+str(int(h))+"\n")
        file.write("p0 = "+str(float(p0))+"\n")
        file.write("pf = "+str(float(pf))+"\n")
        file.write("len_p = "+str(float(len_p))+"\n")
        file.write("systems_per_p = "+str(int(sys_per_p))+"\n")
        file.write("T = "+str(int(T))+"\n")
        file.write("\n")
        file.write("lamb \t F \t std_F\n")
        file.close()
        
    return adresses

def mean_rho(adress, P, p_criticial, net, E, m, h, lista_ps, T, wni, qni, sys_per_p):
    p0, pf, len_p = lista_ps[0], lista_ps[-1], len(lista_ps)
    file = open(adress, 'r')
    file.seek(14)
    ps_generated = int(file.readline().strip())
    file.close()
    N, k = P.shape
    
    if ps_generated < len_p:
        file = open(adress, 'a')
        ps_faltam = lista_ps[ps_generated:]
        
        for i, p in enumerate(tqdm(ps_faltam,colour='red',leave=True)):
            autovals = (1/p_criticial)*p
            try:
                F, F_std = mediador_F_auto(P, p, sys_per_p, T, wni, qni)
            except KeyboardInterrupt:
                file.close()
                file = open(adress, 'r+')
                total_ps = (sum(1 for line in file) - qtde_param)
                file.seek(0)
                file.write(f"ps_generated = {total_ps}\n")
                file.close()
                os.replace(adress, f"MHN/Data/mean_rho/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_sysperp={sys_per_p}_p=({p0},{pf},{total_ps}).txt") # Atualiza a quantidade de sistemas no nome
                raise KeyboardInterrupt
            file.write(str(autovals)+" \t "+str(F)+" \t "+str(F_std)+"\n")
        file.close()
        file = open(adress, 'r+')
        file.write("ps_generated = "+str(int(len_p)))
        file.close()
        os.replace(adress, f"MHN/Data/mean_rho/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_sysperp={sys_per_p}_p=({p0},{pf},{len_p}).txt")
    
    adress = f"MHN/Data/mean_rho/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_sysperp={sys_per_p}_p=({p0},{pf},{len_p}).txt"
    file_read = open(adress, 'r')
    for j in range(qtde_param):
        file_read.readline()
    
    autovals = np.zeros(len_p)
    F = np.zeros(len_p)
    std_F = np.zeros(len_p)
    for j in range(len_p):
        autovals[j], F[j], std_F[j] = np.float64(leitura_ate_tab(file_read.readline()))

    file_read.close()
    return autovals, F, std_F

def mean_plotter(lista_Ns, lista_Es, m, h, lista_ps, T, sys_per_p, mode_lst):
    wni_list, qni_list = lista_Ns//8, lista_Ns//16
    fig, ax = plt.subplots(figsize=(5,4))
    for N, E, wni, qni, mode in zip(lista_Ns, lista_Es, wni_list, qni_list, mode_lst):
        P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
        try:
            adress = file_opener(N, E, m, h, lista_ps[0], lista_ps[-1], len(lista_ps), net, sys_per_p, T)[0]
        except IndexError:
            adress = file_opener(N, E, m, h, lista_ps[0], lista_ps[-1], len(lista_ps), net, sys_per_p, T)[0]
        lamb, F, F_std = mean_rho(adress, P, p_crit, net, E, m, h, lista_ps, T, wni, qni, sys_per_p)
        
        ax.errorbar(lamb, F*1e+5, F_std*1e+5, marker='.', linestyle='none', label=f"N={N}")
    ax.legend(fancybox=True, shadow=True)
    ax.text(0.8, 9.7, 'c',fontsize=20, weight='bold')

    ax.set_xlabel(r"Maior autovalor $\lambda$")
    ax.set_ylabel(r"Atividade média $F (\times 10^{-2} ms^{-1)}$")
    fig.tight_layout()
    
    if len(lista_Ns) == 1:
        N = lista_Ns[0]
        folder_adress = f"MHN/Figs/mean_rho/N={N}"
        
        if os.access(folder_adress, os.F_OK) == False:
            directory = f"N={N}"
            parent_dir = f"MHN/Figs/mean_rho"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
        
        folder_adress += f"/K={E//N}"
        if os.access(folder_adress, os.F_OK) == False:
            directory = f"K={E//N}"
            parent_dir = f"MHN/Figs/mean_rho/N={N}"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)

        folder_adress += f"/m={m}_h={h}"
        if os.access(folder_adress, os.F_OK) == False:
            directory = f"m={m}_h={h}"
            parent_dir = f"MHN/Figs/mean_rho/N={N}/K={E//N}"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)

        fig.savefig(folder_adress + f"/n={net}_p=({lista_ps[0]},{lista_ps[-1]})_sysperp={sys_per_p}.png",dpi=400)
    else:
        folder_adress = f"MHN/Figs/mean_rho/K={E//N}"
        if os.access(folder_adress, os.F_OK) == False:
            directory = f"K={E//N}"
            parent_dir = f"MHN/Figs/mean_rho"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)

        folder_adress += f"/m={m}_h={h}"
        if os.access(folder_adress, os.F_OK) == False:
            directory = f"m={m}_h={h}"
            parent_dir = f"MHN/Figs/mean_rho/K={E//N}"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
            
        fig.savefig(folder_adress + f"/p=({lista_ps[0]},{lista_ps[-1]})_sysperp={sys_per_p}.png",dpi=400)
    return fig, ax

def rho_plotter(N, E, m, h, lista_ps, sys_per_p, mode, T, wni, qni):
    fig, ax = plt.subplots(figsize=(7,4))
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    lista_ps.append(p_crit)
    lamb_base = 1/p_crit
    for p in tqdm(lista_ps):
        for j in range(sys_per_p):
            x_fin, rho = P_time_evaluation(P, p, states, T, wni, qni)
            lamb = p*lamb_base
            if p==p_crit:
                pl, = ax.semilogy(rho, '-', color='red')
            elif p<p_crit:
                pl, = ax.semilogy(rho, '-', alpha=0.6, color='black')
            else:
                pl, = ax.semilogy(rho, '-', alpha=0.6, color='blue')
        pl.set_label(r"$\lambda$"+f" = {round(lamb,5)}")
    ax.legend()
    ax.set_xlabel("t (ms)")
    ax.set_ylabel(r"$\rho$")
    ax.text(-10,0.2,'a',fontsize=20, weight='bold')
    fig.tight_layout()

    folder_adress = f"MHN/Figs/rho_t/N={N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"MHN/Figs/rho_t"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"MHN/Figs/rho_t/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"MHN/Figs/rho_t/N={N}/K={E//N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    fig.savefig(folder_adress + f"/n={net}_p=({lista_ps[0]},{lista_ps[-1]}).png")

states = 5

class graph_view:
    ...
    # if __name__=="__main__":
        # N, K, m, h = 1000, 2, 3, 2
        # E = N*K
        # CIJ = adjacency_maker(N, E, m, h)
        # coon = CIJ_to_coon(CIJ)
        # G = nx.DiGraph()
        # for i in range(len(CIJ)):
        #     G.add_node(i)
        # G.add_edges_from(coon)
        # pos = nx.spring_layout(G)
        # fig, ax = plt.subplots(ncols=2)
        # nx.draw_spring(G, node_size=10, arrowsize=0.01, alpha=0.15, node_color='red', ax=ax[0])
        # ax[1].pcolor(CIJ)
        # plt.show()


class Visual_Iteration:
    ...
    # if __name__=="__main__":
        # CIJ = connectivity_maker(N=1000, E=100000, m=4, h=3, p=0.01212483)
        # P, p = CIJ_to_P(CIJ)
        # P_node_index(P, p, 3, T=100, wni=1000//8, qni=1000//16)
        # plt.show()

class Activity_Rho:
    ...
    # if __name__ == '__main__':
        # N = 10000
        # E, m, h = 12*N, 4, 3
        # lista_ps = [0.09,0.11]
        # T = 700
        # wni, qni = N, N//8
        # rho_plotter(N, E, m, h, lista_ps, 3, 'r1', T, wni, qni)
        # plt.show()

class Phase_Transition:
    ...
    if __name__=="__main__":
        # lista_Ns = np.array([500,750,1000,2500,5000,10000,20000])
        # mode_lst = np.array(['r1','r1','r1','r1','r1','r1','r1'])
        lista_Ns = np.array([500, 10000])
        mode_lst = np.array(['r1','r1'])
        lista_Es, m, h =  lista_Ns*100, 4, 3
        # lista_ps = np.linspace(0.095,0.12,50)
        lista_ps = np.linspace(0.01,0.02,50)
        sys_per_p = 25
        T = 1000
        mean_plotter(lista_Ns, lista_Es, m, h, lista_ps, T, sys_per_p, mode_lst)
        plt.show()
        
