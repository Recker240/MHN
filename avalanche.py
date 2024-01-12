import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from numba import jit

from imports import *
param_qtde = 8

@jit(nopython=True)
def one_time_contador_de_sd(CIJ, states, T):
    N = len(CIJ)
    x = np.zeros(N)
    x[np.random.randint(0,N)] = 1 # Inicialização dos neurônios. Sorteia um autômato e excita-o.
    n_est = 1 # Número de spikes
        
    t=0
    while np.count_nonzero(x) != 0 and t != T:
        x, bool_est = iterator_count(CIJ, x, states)
        n_est += bool_est
        t+=1
    spike = n_est
    duration = t
    return spike, duration

@jit(nopython=True)
def P_one_time_contador_de_sd(P, p, states, T):
    N = len(P)
    x = np.zeros(N)
    x[np.random.randint(0,N)] = 1 # Inicialização dos neurônios. Sorteia um autômato e excita-o.
    n_est = 1 # Número de spikes
        
    t=0
    while np.count_nonzero(x) != 0 and t != T:
        x, bool_est = P_iterator_count(P, x, states, p)
        n_est += bool_est
        t+=1
    spike = n_est
    duration = t
    return spike, duration

@jit(nopython=True)
def mediador_de_s(spikes, duration):
    media_spik = []
    for i, D in enumerate(duration):
        spik_uma_duracao = []
        for j in range(len(spikes)):
            if duration[j] == D:
                spik_uma_duracao.append(spikes[j])
        media_spik.append(mean(spik_uma_duracao))
    return media_spik

@jit(nopython=True)
def acumulator(counts):
    soma = np.zeros(len(counts))
    for i in range(len(counts)):
        soma[i] = np.sum(counts[i:])
    return soma

def file_opener(N,E,m,h,p,net,T, mf):
    folder_adress = f"IC/MHN/Data/{mf}/N={N}"
    
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"IC/MHN/Data/{mf}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Data/{mf}/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        
    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"IC/MHN/Data/{mf}/N={N}/K={E//N}"
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
        if arq[:pos_] == f"n={net}_p={p}_T={T}":
            adresses.append(folder_adress + "/" + arq)
            break
    
    if len(adresses) == 0:
        file = open(folder_adress+f"/n={net}_p={p}_T={T}_sys=0.txt", 'w')
        file.write("systems = 0    \n")
        file.write("N = "+str(int(N))+"\n")
        file.write("E = "+str(int(E))+"\n")
        file.write("m = "+str(int(m))+"\n")
        file.write("h = "+str(int(h))+"\n")
        file.write("p = "+str(float(p))+"\n")
        file.write("T = "+str(int(T))+"\n")
        if mf == 'PS_PD':
            file.write("Tamanho \t Duracao\n")
        elif mf == 'medS_D':
            file.write("Med_spik\n")
        file.close()
        
    return adresses

def Spik_dur(adress, P, E, m, h, p, net, T, desired_sys):
    # Identifica quantos sistemas há no arquivo
    file = open(adress, 'r')
    file.readline(9)
    generated = int(file.readline().strip())
    file.close()
    N, max_k = P.shape
    
    # Se há menos sistemas do que o desejado, roda mais
    if generated < desired_sys:
        file = open(adress, 'a')
        faltam = desired_sys - generated # Quantos sistemas faltam ser rodados
        duration = np.zeros(faltam) # Vetor da duração temporal de cada sistema
        spikes = np.zeros(faltam) # Vetor da quantidade de spikes gerados
        
        for l in tqdm(range(faltam), colour='green',leave=False, desc=f'Simulando sistemas de p={p}'):
            try:
                spikes[l], duration[l] = P_one_time_contador_de_sd(P, p, 3, T)
            except KeyboardInterrupt: # Se o programa teve de ser interrompido, fecha o arquivo, armazenando os dados já processados.
                file.close()
                file = open(adress, 'r+')
                total_sys = sum(1 for line in file) - param_qtde # Quantidade total de sistemas (rodados agora + já tinham)
                file.seek(0)
                file.write(f"systems = {total_sys}\n") # Atualiza a quantidade de sistemas in-file
                file.close()
                os.replace(adress, f"IC/MHN/Data/PS_PD/N={P.shape[0]}/K={E//N}/m={m}_h={h}/n={net}_p={p}_T={T}_sys={total_sys}.txt") # Atualiza a quantidade de sistemas no nome
                raise KeyboardInterrupt
            file.write(str(spikes[l])+" \t "+str(duration[l])+"\n")
        file.close()
        file = open(adress, 'r+')
        total_sys = sum(1 for line in file) - param_qtde  # Quantidade total de sistemas (rodados agora + já tinham)
        file.seek(0)
        file.write(f"systems = {total_sys}") # Atualiza o contador de sistemas in-file
        file.close()
        os.replace(adress, f"IC/MHN/Data/PS_PD/N={P.shape[0]}/K={E//N}/m={m}_h={h}/n={net}_p={p}_T={T}_sys={desired_sys}.txt") # Atualiza o contador de sistemas no nome.
    
    adress = f"IC/MHN/Data/PS_PD/N={P.shape[0]}/K={E//N}/m={m}_h={h}/n={net}_p={p}_T={T}_sys={desired_sys}.txt"
    file_read = open(adress, 'r') # Abre o arquivo com o nome atualizado
    file_read.readline(9)
    generated = int(file_read.readline().strip()) # Obtém a quantidade de sistemas presentes
    spikes = np.zeros(desired_sys)
    duration = np.zeros(desired_sys)
    
    for i in range(param_qtde-1):
        file_read.readline()

    for i in range(desired_sys):
        spikes[i], duration[i] = np.float64(leitura_ate_tab(file_read.readline()))
    file_read.close()
    return spikes, duration

def med_spik_dur(adress, N, E, m, h, p, net, T, spikes, duration):
    systems = len(spikes)
    file = open(adress, 'r')
    generated = sum(1 for line in file) - param_qtde
    file.close()
    if systems != generated:
        file = open(adress, 'r+')
        for i in range(param_qtde):
            file.readline()
        media_spik = mediador_de_s(spikes, duration)
        for i, med in enumerate(media_spik):
            file.write(str(med)+"\n")
        file.seek(0)
        file.write("systems = "+str(systems)+"\n")
        file.close()
        os.replace(adress, f"IC/MHN/Data/medS_D/N={N}/K={E//N}/m={m}_h={h}/n={net}_p={p}_T={T}_sys={systems}.txt")
    
    adress = f"IC/MHN/Data/medS_D/N={N}/K={E//N}/m={m}_h={h}/n={net}_p={p}_T={T}_sys={systems}.txt"
    file = open(adress, 'r')
    med_spik = np.zeros(systems)
    for i in range(param_qtde):
        file.readline()
    for i in range(systems):
        med_spik[i] = float(file.readline())
    file.close()
    return med_spik

def mixed_avalanche_plotter(N, E, m, h, lista_ps, iterate_crit, sys, T, Nhist, mode):
    """_summary_

    Args:
        N (int): Network size
        E (int): Total Edge number
        m (int): Submodule number
        h (int): Hierarchy number
        lista_ps (int): _description_
        sys (int): _description_
        T (int): _description_
        Nhist (int): _description_
        mode (str): File opening mode. values accepted are in the form ```w+r[int]```, with int representing
        which network will be iterated. To not iterate and create new networks, set ```x+r[int]```.

    Returns:
        int: _description_
    """

    fig, ax = plt.subplots(ncols=3, figsize=(11,3.2))
    fig2, ax2 = plt.subplots(ncols=2, figsize=(8.2,3.5))
        
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    lamb_base = 1/p_crit
    if iterate_crit:
        lista_ps.append(p_crit)
        lista_ps = sorted(lista_ps)
        
    for p in tqdm(lista_ps, colour='yellow', leave=True, desc="Plotando os hists..."):
        try:
            P_adress = file_opener(N,E,m,h,p,net,T, "PS_PD")[0]
        except IndexError:
            P_adress = file_opener(N,E,m,h,p,net,T, "PS_PD")[0]
        try:
            mean_adress = file_opener(N,E,m,h,p,net,T, "medS_D")[0]
        except IndexError:
            mean_adress = file_opener(N,E,m,h,p,net,T, "medS_D")[0]
        
        lamb = lamb_base*p
        spikes, durations = Spik_dur(P_adress, P, E, m, h, p, net, T, sys)
        med_spik = med_spik_dur(mean_adress, N, E, m, h, p, net, T, spikes, durations)
                
        counts_spik, bins_spik = np.histogram(spikes, bins=10**np.linspace(np.log10(1), np.log10(max(spikes)), Nhist), density=True)
        counts_dur, bins_dur = np.histogram(durations, bins=10**np.linspace(np.log10(1), np.log10(max(durations)), Nhist), density=True)
        ax[0].loglog(bins_spik[:-1],counts_spik, '.', label=r"$\lambda = $"+ f"{round(lamb,4)}")
        ax[1].loglog(bins_dur[:-1],counts_dur, '.')
        ax[2].loglog(durations, med_spik, '.')
        # ax[0].legend()
        # ax[1].legend()
        # ax[2].legend()
        
        counts_acum_spik, bins_acum_spik = np.histogram(spikes, bins=np.linspace(2, max(spikes), len(spikes)))
        counts_acum_dur, bins_acum_dur = np.histogram(durations, bins=np.linspace(2, max(durations), len(durations)))
        acum_spikes = acumulator(counts_acum_spik)
        acum_dur = acumulator(counts_acum_dur)
        ax2[0].loglog(bins_acum_spik[:-1], acum_spikes, '.', label=r"$\lambda = $"+ f"{round(lamb,4)}")
        ax2[1].loglog(bins_acum_dur[:-1], acum_dur, '.')
        # ax2[0].legend()
        # ax2[1].legend()
    
    x_artificial_spik = np.arange(1,max(spikes))
    x_artificial_dur = np.arange(2,max(durations))
    x_artificial_sd = np.arange(1,max(durations))
    
    ax[0].loglog(x_artificial_spik, np.float_power(x_artificial_spik,-3/2)*0.4,color='brown')
    ax[0].text(1.95e3, 1.4e-9, r"$P(S) \propto S^{- \frac{3}{2}}$", color='brown', fontsize='large')
    ax[1].loglog(x_artificial_dur, np.float_power(x_artificial_dur,-2)*1.9, color='brown')
    ax[1].text(0.8e0, 0.7e-2, r"$P(D) \propto D^{-2}$", color='brown', fontsize='large')
    ax[2].loglog(x_artificial_sd, 0.18*x_artificial_sd**2, color='brown')
    ax[2].text(2.8, 0.3, r"$\langle S _D \rangle \propto D^{2}$", color='brown', fontsize='large')

    # ax[0].text(1e0,1e2,'a',fontsize=20, weight='bold')
    # ax[1].text(1e0,4.1e1,'b',fontsize=20, weight='bold')
    # ax[2].text(1e0,8e5,'c',fontsize=20, weight='bold')
    
    ax2[0].loglog(x_artificial_spik, 3.3e4*np.float_power(x_artificial_spik,-1/2),color='brown')
    ax2[0].text(5e3, 8e2, r"$\zeta(S) \propto S^{- \frac{1}{2}}$", color='brown', fontsize='large')
    ax2[1].loglog(x_artificial_dur, 0.8e5*np.float_power(x_artificial_dur,-1), color='brown')
    ax2[1].text(2e2, 0.6e3, r"$\zeta(D) \propto D^{-1}$", color='brown', fontsize='large')

    # ax2[0].text(1e0,7e4,'d',fontsize=20, weight='bold')
    # ax2[1].text(2e0,8.5e4,'e',fontsize=20, weight='bold')
    fig2.legend(loc='upper center',ncols=5,bbox_to_anchor=(0.5,1.02),fancybox=True,shadow=True)
    fig.legend(loc='upper center',ncols=5,bbox_to_anchor=(0.5,1.02),fancybox=True,shadow=True)
    
    ax[0].set_xlabel("Tamanho S", fontsize=17)
    ax[1].set_xlabel("Duração D", fontsize=17)
    ax[2].set_xlabel("Duração D", fontsize=17)
    ax[0].set_ylabel("P(S)", fontsize=17)
    ax[1].set_ylabel("P(D)", fontsize=17)
    ax[2].set_ylabel(r"$\langle S \rangle$", fontsize=17)

    
    ax2[0].set_xlabel("S", fontsize=17)
    ax2[0].set_ylabel(r"$\zeta (S)$", fontsize=17)
    ax2[1].set_xlabel("D", fontsize=17)
    ax2[1].set_ylabel(r"$\zeta (D)$", fontsize=17)
    ax2[0].set_xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5], [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$',r'$10^5$'], fontsize=12)
    ax2[0].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4], [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$'], fontsize=12)
    ax2[1].set_xticks([1e0, 1e1, 1e2, 1e3], [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$'], fontsize=12)
    ax2[1].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4], [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$'], fontsize=12)

    fig.tight_layout(h_pad=6)
    fig2.tight_layout(h_pad=6)
    
    folder_adress = f"IC/MHN/Figs/avalanche/N={N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"IC/MHN/Figs/avalanche/"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Figs/avalanche/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"IC/MHN/Figs/avalanche/N={N}/K={E//N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    fig.savefig(folder_adress + f"/Hists_T={T}_sys={sys}_n={net}.png") 
    fig2.savefig(folder_adress + f"/Acums_T={T}_sys={sys}_n={net}.png")    
    return fig, ax, fig2, ax2

def Kinouchi_acum_plotter(lista_Ns, lista_Es, m, h, T, sys, Nhist, mode_lst):
    fig2, ax2 = plt.subplots(ncols=2,figsize=(8.2,3.8))
    for N, E, mode in zip(lista_Ns, lista_Es, mode_lst):
        P, p, net = Adjac_file_man(N, E, m, h, mode)
        try:
            P_adress = file_opener(N,E,m,h,p,net,T, "PS_PD")[0]
        except IndexError:
            P_adress = file_opener(N,E,m,h,p,net,T, "PS_PD")[0]
        try:
            mean_adress = file_opener(N,E,m,h,p,net,T, "medS_D")[0]
        except IndexError:
            mean_adress = file_opener(N,E,m,h,p,net,T, "medS_D")[0]
        spikes, durations = Spik_dur(P_adress, P, E, m, h, p, net, T, sys)
    
        counts_acum_spik, bins_acum_spik = np.histogram(spikes, bins=np.linspace(2, max(spikes), len(spikes)))
        counts_acum_dur, bins_acum_dur = np.histogram(durations, bins=np.linspace(2, max(durations), len(durations)))
        acum_spikes = acumulator(counts_acum_spik)
        acum_dur = acumulator(counts_acum_dur)
        ax2[0].loglog(bins_acum_spik[:-1]/N, acum_spikes*(bins_acum_spik[:-1])**(1/2), '.', label=r"$N = $"+ f"{N}")
        ax2[1].loglog(bins_acum_dur[:-1]/(N**(1/2)), acum_dur*(bins_acum_dur[:-1])**(1), '.', label=r"$N = $"+ f"{N}")
        ax2[0].legend()
        ax2[1].legend()
    
    # ax2[0].text(2e-4,5.59e4, 'a', fontweight='bold', fontsize=17)
    # ax2[1].text(2e-2,1.25e5, 'b', fontweight='bold', fontsize=17)
    ax2[0].set_xlabel(r"$\frac{S}{N}$",fontsize=14)
    ax2[0].set_ylabel(r"$\zeta (S) \cdot S^{1/2}$")
    ax2[1].set_xlabel(r"$\frac{D}{N^{1/2}}$",fontsize=14)
    ax2[1].set_ylabel(r"$\zeta (D) \cdot D^{1}$")
    fig2.tight_layout()
    
    folder_adress = f"IC/MHN/Figs/avalanche/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"IC/MHN/Figs/avalanche"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Figs/avalanche/m={m}_h={h}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        
    fig2.savefig(folder_adress + f"/Kin_Acums_n={net}_T={T}_sys={sys}.png",dpi=400)
    return fig2

class Traditional_Hists:
    ...
    N = 10000
    E, m, h = 100*N, 4, 3 
    T = 1000
    lista_ps = [0.009,0.01,0.011,0.013]
    sys = 40000
    Nhist = 200
    mixed_avalanche_plotter(N, E, m, h, lista_ps, True, sys, T, Nhist, 'r1')
    plt.show()

class Kinouchi_Hists:
    ...
    # lista_Ns = np.array([500,750,1000,2500,5000,7500,10000])
    # mode_lst = np.array(['r1','r1','r1','r1','r1','r1','r1'])
    # lista_Es, m, h = 100*lista_Ns, 4, 3
    # systems = 40000
    # T=1000
    # Kinouchi_acum_plotter(lista_Ns, lista_Es, m, h, T, systems, 100, mode_lst)
    # plt.show()

