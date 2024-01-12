import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
from numba import jit
import pandas as pd
import matplotlib.colors as mcolors

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from imports import *

@jit(nopython=True)
def neurons_in_a_module(N, m, k):
    return N/(m**(k))

@jit(nopython=True)
def to_base_10(tup, base):
    num = 0
    for i, el in enumerate(tup):
        num += el*base**(i)
    return int(num)

@jit(nopython=True)
def to_base_m1(num, base):
    quos, rems = [num], []
    i=0
    while True:
        quos.append(int(quos[i]/base))
        rems.append(quos[i]%base)
        if quos[-1] == 0:
            rems.reverse()
            break
        i+=1
    return np.array(rems)

@jit(nopython=True)
def neuron_normalization(N, m, h, neuron):
    factor = lambda k: m**(h-k)
    units = np.array([N/factor(k) for k in range(1,h)])
    locus = neuron/units
    reset_coords = [locus%factor(k) for k in range(1,h)][-1]
    for i, el in enumerate(reset_coords):
        reset_coords[i] = int(el) + 1
    return reset_coords

@jit(nopython=True)
def connection_increaser(N, m, h, s, activated_node):
    s_new = np.copy(s)
    normalized_coords = neuron_normalization(N, m, h, activated_node)
    aux_normalixed_coords = np.copy(normalized_coords)
    for l in range(len(aux_normalixed_coords)+1):
        to_new_base = to_base_10(aux_normalixed_coords, m+1)
        s_new[to_new_base] += 1
        if np.count_nonzero(aux_normalixed_coords) != 0:
            aux_normalixed_coords[l] = 0
    return s_new

@jit(nopython=True)
def modular_initial_position(N, m, h):
    x_inic = np.zeros(N)
    s_inic = np.zeros((m+1)**(h-1))
    raffled_nodes = np.random.randint(0, wni, qni)
    for i in raffled_nodes:
        x_inic[i] = 1
        s_inic = connection_increaser(N, m, h, s_inic, i)
    return x_inic, s_inic

@jit(nopython=True)
def modular_iterator(P, m, h, x, states, p, r):
    N, k = P.shape
    x_new = np.copy(x)
    s = np.zeros((m+1)**(h-1))
    s_new = np.copy(s)

    for i in range(N):
        if x[i] != 0:
            x_new[i] = (x[i]+1)%states
        else:
            stim_test = np.random.uniform(0,1)
            if stim_test <= 1-np.exp(-r*1):
                x_new[i] = 1
                s_new += connection_increaser(N, m, h, s, i)
            else:
                for j in range(k):
                    connected_node = int(P[i,j])
                    if connected_node != -1:
                        u = np.random.uniform(0,1)
                        if x[connected_node] == 1 and u<p:
                            x_new[i] = 1
                            s_new += connection_increaser(N, m, h, s, i)
                            break
                    else:
                        break
    return x_new, s_new

@jit(nopython=True)
def modular_time_iterator(P, m, h, p, states, T, r):
    N = P.shape[0]
    rho = np.zeros((T, (m+1)**(h-1)))
    x, s = modular_initial_position(N, m, h)

    for i in range(T):
        for j in range(len(s)):
            hier = np.count_nonzero(to_base_m1(j,m+1)) # RECHECK THIS, COUNT_NONZERO BY LEN
            nim = neurons_in_a_module(N, m, hier)
            rho[i,j] = s[j]/nim

        x, s = modular_iterator(P, m, h, x, states, p, r)
    return rho

@jit(nopython=True)
def modular_F_dyn(P, m, h, p, systems, states, T, r):
    Fs_per_system = np.zeros((systems, (m+1)**(h-1)))
    for j in range(systems):
        rho = modular_time_iterator(P, m, h, p, states, T, r)
        rho_not_transient = rho[int(0.3*T):,:]
        Fs_per_system[j,:] = matrix_mean(rho_not_transient)
    Fs_per_module = matrix_mean(Fs_per_system)
    stds_Fs_per_module = matrix_std(Fs_per_system, Fs_per_module)
    return Fs_per_module, stds_Fs_per_module

def modular_file_opener(N,E,m,h,p,r0,rf,len_r,T,net):
    folder_adress = f"Data/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/"
    folder_existance_check(folder_adress)
    datas = os.listdir(folder_adress)
    adresses = []
    for arq in datas:
        qtos_ = 0
        for i, carac in enumerate(arq):
            if carac=="_":
                qtos_ += 1
            if qtos_ == 3:
                pos_ = i
                break
        if arq[:pos_] == f"n={net}_T={T}_p={p}":
            adresses.append(folder_adress + arq)
            break
    if len(adresses) == 0:
        file = open(f"Data/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{rf},{len_r}).txt", "w")
        file.write("rs_generated = 0\n")
        file.close()
        adresses = [f"Data/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{rf},{len_r}).txt"]
    return adresses

def modular_file_manager(adress, P, E, m, h, net, p, r0, rf, len_r, T):
    lista_rs = 10**np.linspace(np.log10(r0),np.log10(rf),len_r,endpoint=True)
    file = open(adress, "r")
    rs_generated = int((sum(1 for line in file) - 1)/3)
    file.close()
    N, k = P.shape
    
    if rs_generated < len_r:
        file_app = open(adress, "a")
        rs_faltam = lista_rs[(rs_generated):]

        for r in tqdm(rs_faltam, desc=f"Calculando F para p={p}", colour="yellow", leave=False,miniters=1):
            try:
                pmF, std_pmF = modular_F_dyn(P, m, h, p, systems, states, T, r)
                file_app.write(str(r)+" \n\t")
                file_app.writelines([str(oi)+" \t" for oi in pmF])
                file_app.write("\n\t")
                file_app.writelines([str(oi)+" \t" for oi in std_pmF])
                file_app.write("\n")
            except KeyboardInterrupt:
                file_app.close()
                file_correct = open(adress, "r+")
                file_correct.seek(0)
                total_rs = (sum(1 for line in file_correct) - 1)/3
                print(total_rs)
                file_correct.seek(0)
                file_correct.write("rs_generated = "+str(int(len_r)))
                file_correct.close()
                os.replace(adress, f"Data/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{rf},{total_rs}).txt")
                exit()
        file_app.close()
        file_correct = open(adress, "r+")
        file_correct.write("rs_generated = "+str(int(len_r)))
        file_correct.close()
        os.replace(adress, f"Data/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{rf},{len_r}).txt")
    
    adress = f"Data/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{rf},{len_r}).txt"
    file_read = open(adress, "r")
    
    r = np.zeros(len_r)
    pmF = np.zeros((len_r, (m+1)**(h-1)))
    std_pmF = np.zeros((len_r, (m+1)**(h-1)))
    file_read.readline()
    for j in range(len_r):
        r[j] = np.float64(file_read.readline())
        pmF[j,:] = np.float64(leitura_ate_tab(file_read.readline().strip()))
        std_pmF[j,:] = np.float64(leitura_ate_tab(file_read.readline().strip()))
    file_read.close()

    return r, pmF, std_pmF

def plot_all_F_curves_mpl(N, E, m, h, lista_ps, r0, rf, len_r, iterate_crit, T, mode):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xscale("log")
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    lamb_base = 1/p_crit
    if iterate_crit:
        lista_ps.append(p_crit)

    colormaps = ["Purples", "Blues", "Greens", "Oranges", "Reds",
                      "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
                      "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"]
    Delta = np.zeros(((m+1)**(h-1), len(lista_ps) ))
    sig_Delta = np.zeros(((m+1)**(h-1), len(lista_ps)))

    for i, p in enumerate(tqdm(lista_ps,desc="Calculando as curvas de resposta...",colour="red")):
        try:
            adress = modular_file_opener(N, E, m, h, p, r0, rf, len_r, T, net)[0]
        except IndexError:
            adress = modular_file_opener(N, E, m, h, p, r0, rf, len_r, T, net)[0]
        lamb = lamb_base*p
        r, pmF, std_pmF = modular_file_manager(adress, P, E, m, h, net, p, r0, rf, len_r, T)
        cmap = mpl.colormaps[colormaps[i]]

        for j in range((m+1)**(h-1)):
            F = pmF[:,j]
            std_F = std_pmF[:,j]
            if np.count_nonzero(F) != 0:
                # ax.errorbar(r, F, std_F, linestyle="none", marker=".", color=my_cmap((j-((m+1)**(h-1)))/((m+1)**(h-1))), label=fr"$F_[{np.base_repr(j,m+1)}]$")
                linha, = ax.plot(r, F, marker=".", color=cmap(1 - 0.7*j/((m+1)**(h-1))))
                param = find_F_0(r, F, 1e-3)
                ang, expo, lin = param
                F_0 = 0 if lamb <= 1 else lin
                F_max = 1/states 
                F_tax_inf = F_0 + tax_inf*(F_max - F_0)
                F_tax_sup = F_0 + tax_sup*(F_max - F_0)
                # rmin, F_rmin, rmax, F_rmax, Delta[j,i], sig_Delta[j,i] = find_dyn_range(r, F, F_tax_inf, F_tax_sup)
        linha.set_label(r"$\lambda = $"+ f"{round(lamb,4)}")
    
    # print(pd.DataFrame(Delta, columns=lista_ps, index=[np.base_repr(j,m+1) for j in range((m+1)**(h-1))]))
    ax.set_xlabel(r"$r (ms^{-1})$")
    ax.set_ylabel(r"$F (ms^{-1})$")
    ax.legend(ncol=len(lista_ps))
    fig.tight_layout()

    fig_locator = f"Figs/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/"
    folder_existance_check(fig_locator)
    fig.savefig(fig_locator+f"all_modular_response_n={net}.png")
    return fig

def plot_all_dynamic_curves_mpl(N, E, m, h, lista_ps, r0, rf, len_r, T, mode):
    fig, ax = plt.subplots(figsize=(8,4))
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    lamb_base = 1/p_crit
    lista_ps.append(p_crit)

    Delta = np.zeros(((m+1)**(h-1), len(lista_ps)))
    sig_Delta = np.zeros(((m+1)**(h-1), len(lista_ps)))
    colormaps = ["Purples", "Blues", "Greens", "Oranges", "Reds",
                      "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
                      "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn", "viridis", "plasma", "inferno", "magma", "cividis","ocean", "gist_earth", "terrain", "gnuplot", "gnuplot2", "CMRmap","cubehelix", "brg", "rainbow", "jet",
                      "turbo"]

    for i, p in enumerate(tqdm(lista_ps, desc=f"Calculando curvas de resposta...")):
        try:
            adress = modular_file_opener(N, E, m, h, p, r0, rf, len_r, T, net)[0]
        except IndexError:
            adress = modular_file_opener(N, E, m, h, p, r0, rf, len_r, T, net)[0]
        lamb = lamb_base*p
        r, pmF, std_pmF = modular_file_manager(adress, P, E, m, h, net, p, r0, rf, len_r, T)

        for j in range((m+1)**(h-1)):
            cmap = mpl.colormaps[colormaps[j]]
            F = pmF[:,j]
            std_F = std_pmF[:,j]
            if np.count_nonzero(F) != 0:
                param = find_F_0(r, F, 1e-3)
                ang, expo, lin = param
                F_0 = 0 if lamb <= 1 else lin
                F_max = 1/states 
                F_tax_inf = F_0 + tax_inf*(F_max - F_0)
                F_tax_sup = F_0 + tax_sup*(F_max - F_0)
                rmin, F_rmin, rmax, F_rmax, Delta[j,i], sig_Delta[j,i] = find_dyn_range(r, F, F_tax_inf, F_tax_sup)
                pt, = ax.plot(lamb, Delta[j,i], ".", color=cmap(1 - 0.7*j/((m+1)**(h-1))))
                pt.set_label(to_base_m1(j,m+1)) if i==0 else ...
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta (dB)$")
    ax.legend(ncol=h+1)
    fig.tight_layout()

    fig_locator = f"Figs/modular_dynamic/N={N}/K={E//N}/m={m}_h={h}/"
    folder_existance_check(fig_locator)
    fig.savefig(fig_locator+f"all_modular_dynamic_n={net}.png")
    return fig

def plot_all_F_curves_str(N, E, m, h, r0, rf, len_r, T, mode, load):
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    print(f"Cálculos para rede n={net}")
    num_modules = (m+1)**(h-1)

    Delta = np.zeros(num_modules)
    sig_Delta = np.zeros(num_modules)
    fig = make_subplots(cols=1,rows=1)
    try:
        adress = modular_file_opener(N, E, m, h, p_crit, r0, rf, len_r, T, net)[0]
    except IndexError:
        adress = modular_file_opener(N, E, m, h, p_crit, r0, rf, len_r, T, net)[0]
    
    r, pmF, std_pmF = modular_file_manager(adress, P, E, m, h, net, p_crit, r0, rf, len_r, T)

    for j in range(num_modules):
        F = pmF[:,j]
        std_F = std_pmF[:,j]
        if np.count_nonzero(F) != 0:
            fig.add_trace(go.Scatter(mode="markers",x=r, y=F, error_y=dict(type="data",array=std_F,visible=False),name=f"{np.base_repr(j,m+1)}"), 1,1)
            F_0 = 0
            F_max = 1/states
            F_tax_inf = F_0 + tax_inf*(F_max - F_0)
            F_tax_sup = F_0 + tax_sup*(F_max - F_0)
            rmin, F_rmin, rmax, F_rmax, Delta[j], sig_Delta[j] = find_dyn_range(r, F, F_tax_inf, F_tax_sup)
    

    indexes_enuples = [str(to_base_m1(j,m+1)) for j in range(num_modules)]
    degrees = [np.count_nonzero(to_base_m1(j,m+1)) for j in range(num_modules)]
    Delta_df = pd.DataFrame(np.matrix.transpose(np.array([Delta, sig_Delta, indexes_enuples, degrees])), columns=["Delta", "Erro", "Módulo", "Grau Hierárquico"])
    Delta_df["Delta"] = pd.to_numeric(Delta_df["Delta"])
    Delta_df["Erro"] = pd.to_numeric(Delta_df["Erro"])
    Delta_df["Grau Hierárquico"] = pd.to_numeric(Delta_df["Grau Hierárquico"])

    if load:
        fig.update_xaxes(type="log",row=1,col=1)
        fig.update_layout(xaxis_title="r(kHz)",yaxis_title="F (kHz)")
        fig2 = go.Figure(data=go.Scatter(mode="markers", x=Delta_df["Grau Hierárquico"], y=Delta_df["Delta"], text=Delta_df["Módulo"]),layout_yaxis_range=[24.5,27.5])
        fig3 = make_subplots()
        for i in range(1,h):
            mask = (Delta_df["Grau Hierárquico"]==i) & (Delta_df["Delta"] > 1)
            masked_Deltas = np.array(Delta_df[mask]["Delta"])
            len_masked_Deltas = len(masked_Deltas)
            diffs_hist, diffs_bins = np.histogram(masked_Deltas - Delta[0], density=True, bins=len_masked_Deltas)
            fig3.add_trace(go.Scatter(mode="markers", x=diffs_bins, y=diffs_hist, name=f"grau {i}", text=f"D = {masked_Deltas[len_masked_Deltas//2]}"))

        st.text(f"Curvas de resposta e de faixa dinâmica para N={N}, K={E//N}, m={m} e h={h}:")
        st.write(Delta_df)
        # st.write("Curvas de resposta para cada módulo")
        # st.plotly_chart(fig, use_container_width=True)
        st.write("Faixas dinâmicas de cada grau hierárquico")
        st.plotly_chart(fig2, use_container_width=True)
        st.latex(r"Diferença de cada \Delta para o valor da rede toda")
        st.plotly_chart(fig3, use_container_width=True)
    return fig

        
def mh_finder(selected_mh):
    for l in range(len(selected_mh)-1,-1,-1):
        if selected_mh[l] == '=':
            h = int(selected_mh[(l+1):])
            break
    for i in range(len(selected_mh)):
        aux = selected_mh[i]
        if selected_mh[i] == '=':
            for j in range(i,len(selected_mh), 1):
                aux2 = selected_mh[j]
                if selected_mh[j] == '_':
                    m = selected_mh[(i+1):j]
                    return int(m), int(h)


available_Ns = os.listdir("Data/modular_dynamic/")
selected_N = st.selectbox("Selecione o Tamanho desejado para a rede", available_Ns)
available_Ks = os.listdir("Data/modular_dynamic/"+selected_N+"/")
selected_K = st.selectbox("Selecione o grau médio desejado da rede", available_Ks)
available_mh = os.listdir("Data/modular_dynamic/"+selected_N+"/"+selected_K+"/")
selected_mh = st.selectbox("Selecione o padrão hierárquico desejado da rede", available_mh)

folder_adress = "Data/modular_dynamic/"+selected_N+"/"+selected_K+"/"+selected_mh
net = os.listdir(folder_adress)[0][2]

N = int(selected_N[2:])
E = int(selected_K[2:])*N
m, h = mh_finder(selected_mh)

# Globals
wni, qni = N, N//10
systems, states = 30, 5
tax_inf, tax_sup = 0.1, 0.9
T = 1000
fig = plot_all_F_curves_str(N, E, m, h, 1e-5, 1e+2, 50, T, 'r'+str(net), True)

