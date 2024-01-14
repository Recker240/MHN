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
import plotly.express as px
import plotly.figure_factory as ff
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
    difference = [Delta[j] - Delta[0] for j in range(len(Delta))]
    Delta_df = pd.DataFrame(np.matrix.transpose(np.array([Delta, sig_Delta, indexes_enuples, degrees, difference])), columns=["Delta", "Erro", "Modularização", "Grau Hierárquico", "Diferença"])
    Delta_df["Delta"] = pd.to_numeric(Delta_df["Delta"])
    Delta_df["Erro"] = pd.to_numeric(Delta_df["Erro"])
    Delta_df["Grau Hierárquico"] = pd.to_numeric(Delta_df["Grau Hierárquico"])
    Delta_df["Diferença"] = pd.to_numeric(Delta_df["Diferença"])
    Delta_df = Delta_df[Delta_df["Delta"] > 1]

    if load:
        fig.update_xaxes(type="log",row=1,col=1)
        fig.update_layout(xaxis_title="r(kHz)",yaxis_title="F (kHz)")
        fig2 = px.scatter(Delta_df, x="Grau Hierárquico", y="Delta", color="Grau Hierárquico", hover_data="Modularização")
        fig2.update_yaxes(range=[0.99*min(Delta_df["Delta"]),1.01*max(Delta_df["Delta"])])
        fig2.update_xaxes(showgrid=True)
        fig2.update_layout(hovermode="y")
        diffs_Deltas = []
        modularization_specific_deg = []
        for i in range(1,h):
            mask = (Delta_df["Grau Hierárquico"]==i)
            masked_Deltas = np.array(Delta_df[mask]["Delta"])
            diffs_Deltas.append(masked_Deltas)
            modularization_specific_deg.append(Delta_df[mask]["Modularização"])

        fig3 = ff.create_distplot(diffs_Deltas, [f"Grau hierárquico {i}" for i in range(1,h)], bin_size=0.2, rug_text=[modularization_specific_deg[i] for i in range(len(diffs_Deltas))])
        fig3.add_vline(x=Delta[0], line_dash='dash', line_width=1, line=dict(color='White',))
        fig3.update_layout(xaxis_title="Valores de faixa dinâmica (dB)", yaxis_title='Densidade')

        cols = st.columns(2)
        with cols[0]:
            st.write(Delta_df)
        with cols[1]:
            response = st.checkbox("Exibir Curvas de resposta para cada módulo (Pode levar alguns minutos)")
            if response:
                st.plotly_chart(fig, use_container_width=True)
        cols = st.columns(2)
        with cols[0]:
            st.write("Faixas dinâmicas de cada grau hierárquico")
            st.plotly_chart(fig2, use_container_width=True)
        with cols[1]:
            st.write("Histograma normalizado dos valores de faixa dinâmica. A linha tracejada representa o valor da rede toda.")
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

st.set_page_config(layout="wide")
st.write("Selecione os parâmetros desejados da rede (Obs.: Os valores de N=10125 são os mais robustos e com maior qualidade):")
cols = st.columns(4)
available_Ns = sorted(os.listdir("Data/modular_dynamic/"))
available_Ns.sort(key=len)
available_Ns.reverse()
with cols[0]:
    selected_N = st.selectbox("Número de nós", available_Ns)
available_Ks = sorted(os.listdir("Data/modular_dynamic/"+selected_N+"/"))
available_Ks.sort(key=len)
available_Ks.reverse()
with cols[1]:
    selected_K = st.selectbox("Grau médio", available_Ks)
available_mh = os.listdir("Data/modular_dynamic/"+selected_N+"/"+selected_K+"/")
available_mh.sort(key=len)
available_mh.reverse()
with cols[2]:
    selected_mh = st.selectbox("Padrão hierárquico", available_mh)

folder_adress = "Data/modular_dynamic/"+selected_N+"/"+selected_K+"/"+selected_mh
possible_data = os.listdir(folder_adress)
nets = []
for data in possible_data:
    for index, letra in enumerate(data):
        if letra == '_':
            nets.append(data[:index])
            break

with cols[3]:
    selected_n = st.selectbox("Selecione uma matriz de adjacências específica:", nets)

N = int(selected_N[2:])
E = int(selected_K[2:])*N
m, h = mh_finder(selected_mh)
n = selected_n[2:]

st.write(f"Nesse caso, há módulos de tamanhos {[N/(m**(h-k)) for k in range(1,h)]}.")

# Globals
wni, qni = N, N//10
systems, states = 30, 5
tax_inf, tax_sup = 0.1, 0.9
T = 1000
fig = plot_all_F_curves_str(N, E, m, h, 1e-5, 1e+2, 50, T, 'r'+str(n), True)
