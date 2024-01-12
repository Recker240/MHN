import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
from numba import jit
import statsmodels.api as sm
from scipy.optimize import curve_fit
from matplotlib.patches import FancyArrowPatch, ArrowStyle
import pandas as pd

from imports import P_time_evaluation_external, Adjac_file_man, degree_calc
from classical import mean, leitura_ate_tab

qtde_param = 12

@jit(nopython=True)
def mediador_F_auto_dyn(P, p, systems, T, wni, qni, r):
    F_list = np.zeros(systems)
    rho_std, F_std = 0, 0
    for j in range(systems):
        x_fin, rho = P_time_evaluation_external(P, p, states, T, wni, qni, r)
        rho_not_trans = rho[int(0.3*len(rho)):]
        F_list[j] = mean(rho_not_trans)/T
    F = np.mean(F_list)
    F_std = np.std(F_list)
    return F, F_std

@jit(nopython=True)
def find_intersection(r_adjust, F_adjust, y_des):
    xa, xb = r_adjust[0], r_adjust[1]
    ya, yb = F_adjust[0], F_adjust[1]
    b = (yb*np.log(xa) - ya*np.log(xb))/np.log(xa/xb)
    a = (ya - yb)/np.log(xa/xb)
    x_des = np.exp((y_des-b)/a)
    return x_des, y_des

@jit(nopython=True)
def find_dyn_range(lista_rs, lista_Fs, F_tax_inf, F_tax_sup):
    rmin_adjust = np.zeros(2)
    rmax_adjust = np.zeros(2)
    F_rmin_adjust = np.zeros(2)
    F_rmax_adjust = np.zeros(2)
    
    for i in range(len(lista_rs)):
        if lista_Fs[i] > F_tax_inf:
            rmin_adjust[0] = lista_rs[i-1]
            rmin_adjust[1] = lista_rs[i]
            F_rmin_adjust[0] = lista_Fs[i-1]
            F_rmin_adjust[1] = lista_Fs[i]
            break
    list(lista_Fs).reverse()
    for i in range(len(lista_Fs)):
        if lista_Fs[i] > F_tax_sup:
            rmax_adjust[0] = lista_rs[i-1]
            rmax_adjust[1] = lista_rs[i]
            F_rmax_adjust[0] = lista_Fs[i-1]
            F_rmax_adjust[1] = lista_Fs[i]
            break
    
    rmin, F_rmin = find_intersection(rmin_adjust, F_rmin_adjust, F_tax_inf)
    rmax, F_rmax = find_intersection(rmax_adjust, F_rmax_adjust, F_tax_sup)
    sig_rmin = - abs((rmin_adjust[1] - rmin_adjust[0])/np.sqrt(2)) + np.sqrt((rmin-rmin_adjust[1])**2 + (rmin-rmin_adjust[0])**2)
    sig_rmax = - abs((rmax_adjust[1] - rmax_adjust[0])/np.sqrt(2)) + np.sqrt((rmax-rmax_adjust[1])**2 + (rmax-rmax_adjust[0])**2)

    Delta = 10*np.log10(rmax/rmin)
    sig_Delta = (10/np.log(10)) * np.sqrt((sig_rmax**2 / rmax**2 + sig_rmin**2 / rmin**2))
    return rmin, F_rmin, rmax, F_rmax, Delta, sig_Delta

@jit(nopython=True)
def theoretical_dynamic(y, x, F_max, F_0, K, lamb, n):
    num = np.log((1-n*x*F_max - n*F_0 + n*x*F_0)/(1-(n-1)*x*F_max - (n-1)*F_0 + (n-1)*x*F_0)) - K*np.log(1 - (lamb/K)*(F_0 + x*(F_max - F_0)))
    den = np.log((1-n*y*F_max - n*F_0 + n*y*F_0)/(1-(n-1)*y*F_max - (n-1)*F_0 + (n-1)*y*F_0)) - K*np.log(1 - (lamb/K)*(F_0 + y*(F_max - F_0)))

    return 10*np.log10(num/den)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
def find_F_0(lista_rs, lista_Fs, lim):
    rs_adjust = lista_rs[lista_rs < lim]
    Fs_adjust = lista_Fs[:len(rs_adjust)]
    def poly(x, a, b, c):
        return a*x** b + c
    param, _ = curve_fit(poly, rs_adjust, Fs_adjust, maxfev=1500)
    return param

def file_opener_dyn(N,E,m,h,p,r0,rf,len_r,T,net):
    folder_adress = f"IC/MHN/Data/dynamic/N={N}"
    
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"IC/MHN/Data/dynamic"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Data/dynamic/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"IC/MHN/Data/dynamic/N={N}/K={E//N}"
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
        if arq[:pos_] == f"n={net}_T={T}_p={p}":
            adresses.append(folder_adress + "/" + arq)
            break
    
    if len(adresses) == 0:
        file = open(folder_adress+f"/n={net}_T={T}_p={p}_r=({r0},{rf},0).txt", 'w')
        file.write("rs_generated = 0   \n")
        file.write("N = "+str(int(N))+"\n")
        file.write("E = "+str(int(E))+"\n")
        file.write("m = "+str(int(m))+"\n")
        file.write("h = "+str(int(h))+"\n")
        file.write("p = "+str(int(p))+"\n")
        file.write("r0 = "+str(float(r0))+"\n")
        file.write("rf = "+str(float(rf))+"\n")
        file.write("len_r = "+str(float(len_r))+"\n")
        file.write("T = "+str(int(T))+"\n")
        file.write("\n")
        file.write("r \t F \t std_F\n")
        file.close()
        
    return adresses

def resposta_files(adress, P, E, m, h, net, p, r0, rf, len_r, T, wni, qni):
    lista_rs = 10**np.linspace(np.log10(r0),np.log10(rf),len_r,endpoint=True)
    file = open(adress, 'r')
    file.seek(14)
    rs_generated = int(file.readline().strip())
    file.close()
    N, k = P.shape
    
    if rs_generated < len_r:
        file = open(adress, 'a')
        rs_faltam = lista_rs[(rs_generated):]
        
        for r in tqdm(rs_faltam,colour='yellow',leave=True,desc=f'Calculando F para m={m}, h={h} e p={p}...',miniters=1):
            try:
                F, F_std = mediador_F_auto_dyn(P, p, sys_per_p, T, wni, qni, r)
                file.write(str(r)+" \t "+str(F)+" \t "+str(F_std)+"\n")
            except KeyboardInterrupt:
                file.close()
                file = open(adress, 'r+')
                total_rs = (sum(1 for line in file) - qtde_param) - 1
                file.seek(0)
                file.write(f"rs_generated = {total_rs}\n")
                file.close()
                os.replace(adress, f"IC/MHN/Data/dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{r},{len_r}).txt") # Atualiza a quantidade de sistemas no nome
                raise KeyboardInterrupt
        file.close()
        file = open(adress, 'r+')
        file.write("rs_generated = "+str(int(len_r)))
        file.close()
        os.replace(adress, f"IC/MHN/Data/dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{rf},{len_r}).txt")
    
    adress = f"IC/MHN/Data/dynamic/N={N}/K={E//N}/m={m}_h={h}/n={net}_T={T}_p={p}_r=({r0},{rf},{len_r}).txt"
    file_read = open(adress, 'r')
    for j in range(qtde_param):
        file_read.readline()
    
    r = np.zeros(len_r)
    F = np.zeros(len_r)
    std_F = np.zeros(len_r)
    for j in range(len_r):
        r[j], F[j], std_F[j] = np.float64(leitura_ate_tab(file_read.readline()))

    file_read.close()
    return r, F, std_F

def plot_delta_h(N, E, m, h, lista_ps, r0, rf, len_r, iterate_crit, T, wni, qni, mode):
    fig, ax = plt.subplots(figsize=(4.5,4))
    ax.set_xscale('log')
    ax.set_yscale('log')
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    lamb_base = 1/p_crit
    if iterate_crit:
        lista_ps.append(p_crit)
        lista_ps = sorted(lista_ps)
        
    for p in lista_ps:
        try:
            adress = file_opener_dyn(N,E,m,h,p,r0, rf, len_r,T,net)[0]
        except IndexError:
            adress = file_opener_dyn(N,E,m,h,p,r0, rf, len_r,T,net)[0]
        lamb = lamb_base*p
        r, F, std_F = resposta_files(adress, P, E, m, h, net, p, r0, rf, len_r, T, wni, qni)
        F *= 1e+3
        if p == p_crit:
            cr = ax.errorbar(r, F, std_F, linestyle='none', marker='.', color='red')
        elif p<p_crit:
            subc = ax.errorbar(r, F, std_F, linestyle='none', marker='.', color='black')
        else:
            supc = ax.errorbar(r, F, std_F, linestyle='none', marker='.', color='blue')
    
    cr.set_label("Caso crítico")
    supc.set_label("Caso supercrítico")
    subc.set_label("Caso subcrítico")
    r_artificial = np.linspace(0.5e-4,0.5e-3,100)
    ax.loglog(r_artificial, 0.6*np.power(r_artificial,1/2),color='red')
    ax.text(4e-6, 3e-3, r"$F \propto r^{\frac{1}{2}}$", color='red',fontsize='large')

    r_artificial = np.linspace(1e-5,1e-4,100)
    ax.loglog(r_artificial, 2*np.power(r_artificial,1),color='black')
    ax.text(2.3e-5, 2e-5, r"$F \propto r^{1}$", color='black',fontsize='large')
    # ax.text(0.55e-6,4.1e-1, 'c', fontweight='bold', fontsize=20)
    ax.set_xlabel(r'$r (ms^{-1})$')
    ax.set_ylabel(r'$F (ms^{-1})$')
    ax.legend()
    
    fig.tight_layout()
    folder_adress = f"IC/MHN/Figs/dynamic/N={N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"IC/MHN/Figs/dynamic"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Figs/dynamic/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"IC/MHN/Figs/dynamic/N={N}/K={E//N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    fig.savefig(f"IC/MHN/Figs/dynamic/N={N}/K={E//N}/m={m}_h={h}/delta_h_n={net}.png", dpi=400)
    return fig

def plot_resposta(N, E, m, h, lista_ps, r0, rf, len_r, iterate_crit, T, wni, qni, mode):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_xscale('log')
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    lamb_base = 1/p_crit
    if iterate_crit:
        lista_ps.append(p_crit)
        # lista_ps = sorted(lista_ps)
        
    for p in lista_ps:
        try:
            adress = file_opener_dyn(N,E,m,h,p,r0, rf, len_r,T,net)[0]
        except IndexError:
            adress = file_opener_dyn(N,E,m,h,p,r0, rf, len_r,T,net)[0]
        lamb = lamb_base*p
        r, F, std_F = resposta_files(adress, P, E, m, h, net, p, r0, rf, len_r, T, wni, qni)
        F *= 1e+3
        if p == p_crit:
            cr = ax.errorbar(r, F, std_F, linestyle='none', marker='.',color='red')
        elif p<p_crit:
            subc = ax.errorbar(r, F, std_F, linestyle='none', marker='.',color='black', alpha=0.8)
        else:
            supc = ax.errorbar(r, F, std_F, linestyle='none', marker='.', color='blue', alpha=0.8)
        F_artificial = np.linspace(0.932*min(F),5*max(F),1000)
        # relacao_teorica = k*np.log(1-sig*F_artificial/k) + np.log(1-(states-1)*F_artificial) - np.log(1-states*F_artificial)
        param = find_F_0(r, F,1e-3) 
        ang, expo, lin = param
        print(expo)
        F_0 = 0 if lamb <= 1 else lin
        # F_0 = 0 if lamb <= 1 else F[0]
        F_max = 1/states 
        F_tax_inf = F_0 + tax_inf*(F_max - F_0)
        F_tax_sup = F_0 + tax_sup*(F_max - F_0)
        rmin, F_rmin, rmax, F_rmax, Delta, sig_Delta = find_dyn_range(r, F, F_tax_inf, F_tax_sup)
        r_artificial = 10**np.linspace(np.log10(1e-7),np.log10(1e-3),40)
        print(p, Delta)
        # ax.plot(r_artificial, ang*r_artificial**expo + lin, '-')
        # ax.plot(1e-7,lin,'y.')
    ax.set_xlabel(r'$r (ms^{-1})$')
    ax.set_ylabel(r'$F (ms^{-1})$')
    if len(lista_ps) != 1:
        cr.set_label("Caso crítico")
        subc.set_label('Casos Subcríticos')
        supc.set_label('Casos Supercríticos')
        ax.legend()
        # ax.text(0.8e-6,0.215, 'a', fontweight='bold', fontsize=20)
    else:
        ax.set_ylim(ax.get_ylim())
        ax.set_xlim(ax.get_xlim())
        ax.plot(rmin, F_rmin, 'k.', rmax, F_rmax, 'k.', alpha=0.4)
        ax.plot([rmin, rmin], [ax.get_ylim()[0],F_rmin], 'k-', [rmax, rmax],[ax.get_ylim()[0],F_rmax], 'k-',linewidth=0.7)
        ax.plot([rmin,ax.get_xlim()[0]], [F_rmin,F_rmin], 'k-', [rmax,ax.get_xlim()[0]], [F_rmax,F_rmax], 'k-', linewidth=0.7)
        ax.annotate(r'$(r_{0.1}, F_{0.1})$',xy=(rmin, F_rmin), xytext=(1e-4,0.05), arrowprops=dict(facecolor='black',width=0.6, headlength=3.5, headwidth=4.5))
        ax.annotate(r'$(r_{0.9}, F_{0.9})$',xy=(rmax, F_rmax), xytext=(1e-2,0.15), arrowprops=dict(facecolor='black',width=0.6, headlength=3.5, headwidth=4.5))
        style = ArrowStyle('<->', head_length=3.5, head_width=1.5)
        arrow = FancyArrowPatch((rmin, 0.015),(rmax, 0.015), color='black', linewidth=1, arrowstyle=style)
        ax.add_patch(arrow)
        ax.text(3e-3, 0, r'$\Delta = $'+f'{round(Delta,3)}'+r'$\pm$'+f'{round(sig_Delta,3)}'+r'$dB$')
        ax.text(0.8e-6,0.215, 'b', fontweight='bold', fontsize=20)
    
    fig.tight_layout()
    folder_adress = f"IC/MHN/Figs/dynamic/N={N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"IC/MHN/Figs/dynamic"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Figs/dynamic/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"IC/MHN/Figs/dynamic/N={N}/K={E//N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    if len(lista_ps) > 1:
        fig.savefig(f"IC/MHN/Figs/dynamic/N={N}/K={E//N}/m={m}_h={h}/resposta_variosp_n={net}.png", dpi=400)
    else:
        fig.savefig(f"IC/MHN/Figs/dynamic/N={N}/K={E//N}/m={m}_h={h}/p={lista_ps[0]}_n={net}.png", dpi=400)
    return fig

def plot_Dynamic(N, E, m, h, lista_ps, r0, rf, len_r, T, wni, qni, mode):
    fig, ax = plt.subplots()
    P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    K = degree_calc(P)
    lamb_base = 1/p_crit
    lista_ps.append(p_crit)
    lista_ps = sorted(lista_ps)

    lamb_lst = np.zeros_like(lista_ps)
    
    for i, p in enumerate(lista_ps):
        try:
            adress = file_opener_dyn(N,E,m,h,p,r0, rf, len_r,T,net)[0]
        except IndexError:
            adress = file_opener_dyn(N,E,m,h,p,r0, rf, len_r,T,net)[0]
        lamb_lst[i] = lamb_base*p
        lista_rs, lista_Fs, std_F = resposta_files(adress, P, E, m, h, net, p, r0, rf, len_r, T, wni, qni)
        lista_Fs *= 1e+3
        ang, expo, lin = find_F_0(lista_rs, lista_Fs, 1e-3)
        F_0 = 0 if lamb_lst[i] <= 1 else lin
        # F_0 = 0 if lamb <= 1 else F[0]
        F_max = 1/states 
        F_tax_inf = F_0 + tax_inf*(F_max - F_0)
        F_tax_sup = F_0 + tax_sup*(F_max - F_0)
        rmin, F_rmin, rmax, F_rmax, Delta, sig_Delta = find_dyn_range(lista_rs, lista_Fs, F_tax_inf, F_tax_sup)
        print(p, Delta)
        if p == p_crit:
            cr = ax.errorbar(lamb_lst[i], Delta, sig_Delta, linestyle='none', marker='.', color='red')
        elif p<p_crit:
            subc = ax.errorbar(lamb_lst[i], Delta, sig_Delta, linestyle='none', marker='.', color='black')
        elif p>p_crit:
            supc = ax.errorbar(lamb_lst[i], Delta, sig_Delta, linestyle='none', marker='.', color='blue')
    
    # theo = theoretical_dynamic(tax_inf, tax_sup, F_max, F_0, K, lamb_lst, states)
    # ax.plot(lamb_lst, theo)

    cr.set_label('Caso crítico')
    subc.set_label('Caso subcrítico')
    supc.set_label('Caso supercrítico')
    ax.text(0.57,27.1, 'd', fontweight='bold', fontsize=20)
    ax.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta (dB)$")

    folder_adress = f"IC/MHN/Figs/dynamic/N={N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"IC/MHN/Figs/dynamic"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Data/dynamic/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    folder_adress += f"/m={m}_h={h}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"m={m}_h={h}"
        parent_dir = f"IC/MHN/Figs/dynamic/N={N}/K={E//N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    fig.savefig(folder_adress + f"/Deltaplot_n={net}.png", dpi=400)
    return fig

def plot_pcolor_varying_hm_critical_p_rect_grid(N, E, lista_ms, r0, rf, len_r, T, wni, qni, mode):
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    my_cmap = mpl.colormaps['gnuplot']
    fig2, ax2 = plt.subplots(figsize=(4,4))

    hmax = int(np.log(N)/np.log(lista_ms[0]) + 1)
    Delta_matrix = np.zeros((hmax-2,len(lista_ms)))
    sig_Delta_matrix = np.zeros_like(Delta_matrix)
    
    for i, m in enumerate(tqdm(lista_ms,desc="Setting up networks...",colour='red',leave=False)):
        lista_hs = np.arange(2,int(np.log(N)/np.log(m) + 1))
        for j, h in enumerate(lista_hs):
            try:
                P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
            except FileNotFoundError:
                P, p_crit, net = Adjac_file_man(N, E, m, h, 'w')
                mode = 'r1'
                P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
    
    for i, m in enumerate(lista_ms):
        lista_hs = np.arange(2,int(np.log(N)/np.log(m) + 1))
        for j, h in enumerate(lista_hs):
            P, p_crit, net = Adjac_file_man(N, E, m, h, mode)
            try:
                adress = file_opener_dyn(N,E,m,h,p_crit,r0,rf,len_r,T,net)[0]
            except IndexError:
                adress = file_opener_dyn(N,E,m,h,p_crit,r0, rf, len_r,T,net)[0]
            lista_rs, lista_Fs, std_F = resposta_files(adress, P, E, m, h, net, p_crit, r0, rf, len_r, T, wni, qni)
            lista_Fs *= 1e+3
            F_0 = 0
            F_max = 1/states
            F_tax_inf = F_0 + tax_inf*(F_max - F_0)
            F_tax_sup = F_0 + tax_sup*(F_max - F_0)
            rmin, F_rmin, rmax, F_rmax, Delta, sig_Delta = find_dyn_range(lista_rs, lista_Fs, F_tax_inf, F_tax_sup) 
            Delta_matrix[j,i] = Delta
            sig_Delta_matrix[j,i] = sig_Delta
            ax.errorbar(m, h, Delta, zerr=sig_Delta, marker='.', linestyle='none', c=my_cmap((Delta_matrix[j,i]-25.7)/(26.5-25.7)))
    
    lista_hs_max = np.arange(2,int(np.log(N)/np.log(lista_ms[0]) + 1))
    Delta_matrix[Delta_matrix == 0] = np.nan
    pdDelta = pd.DataFrame(Delta_matrix,index=[f'h = {i}' for i in lista_hs_max],columns=[f'm = {i}' for i in lista_ms])
    X, Y = np.meshgrid(lista_ms, lista_hs_max)
    im = ax2.pcolormesh(X, Y, pdDelta, cmap=my_cmap, vmin=25.7,vmax=26.5)
    fig2.colorbar(im, ax=ax2, cmap=my_cmap)
    print(pdDelta)

    # ax.plot_surface(X, Y, pdDelta, rstride=1,  cstride=1, cmap=my_cmap,vmin=25.7,vmax=26.5,zsort='max')
    ax.set_xlabel('m',fontsize=17)
    ax.set_ylabel('h',fontsize=17)
    ax.set_zlabel(r'$\Delta (dB)$')
    ax.set_xticks(lista_ms)
    ax.set_yticks(lista_hs_max)
    ax.set_zlim((25,27))
    # fig.text(0.1,0.9, 'b', fontweight='bold', fontsize=20)

    ax2.set_xlabel("m",fontsize=17)
    ax2.set_ylabel("h",fontsize=17)
    ax2.set_xticks(lista_ms)
    ax2.set_yticks(lista_hs_max)
    # ax2.text(1.75,13.75, 'a', fontweight='bold', fontsize=20)

    fig.tight_layout()
    fig2.tight_layout()
    folder_adress = f"IC/MHN/Figs/dynamic/N={N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"N={N}"
        parent_dir = f"IC/MHN/Figs/dynamic"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    folder_adress += f"/K={E//N}"
    if os.access(folder_adress, os.F_OK) == False:
        directory = f"K={E//N}"
        parent_dir = f"IC/MHN/Data/dynamic/N={N}"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    fig.savefig(folder_adress + f"/Delta_mh_3d_plot_n={net}.png", dpi=400)
    fig2.savefig(folder_adress + f"/Delta_mh_pcolor_n={net}.png", dpi=400)
    return fig

# Global Variables
tax_inf, tax_sup = 0.1,0.9 
sys_per_p = 25
states = 5

if __name__=="__main__":
    class delta_h_plotter:
        ...
        # N = 10000
        # E, m, h =  N*100, 4, 3
        # T = 1000
        # r0, rf, len_r = 1e-6, 1e+2, 50
        # # lista_ps = list(np.round(np.arange(0.075,0.14,0.0005),6))
        # # lista_ps.remove(0.1)
        # lista_ps = list(np.round(np.arange(0.009,0.018,0.0005),6))
        # lista_ps.remove(0.012)
        # lista_ps.remove(0.0125)
        # plot_delta_h(N, E, m, h, lista_ps, r0, rf, len_r, True, T, N//8, N//16, 'r1')
        # plt.show()

    class resposta_plotter:
        ...
        N = 10000
        E, m, h =  N*100, 4, 3
        T = 1000
        r0, rf, len_r = 1e-6, 1e+2, 50  
        lista_ps = list(np.round(np.arange(0.009,0.018,0.0005),6))
        lista_ps.remove(0.012)
        # lista_ps.append(0.01225)
        lista_ps = sorted(lista_ps)
        # lista_ps = [0.01225]
        plot_resposta(N, E, m, h, lista_ps, r0, rf, len_r, True, T, N//8, N//16, 'w')
        plt.show()
        
    class Delta_lamb_plotter:
        ...
        # N = 10000
        # E, m, h =  N*100, 4, 3
        # T = 1000
        # r0, rf, len_r = 1e-6, 1e+2, 50
        # lista_ps = list(np.round(np.arange(0.007,0.018,0.0005),6))
        # lista_ps.remove(0.012)
        # lista_ps.append(0.01225)
        # lista_ps = sorted(lista_ps)
        # plot_Dynamic(N, E, m, h, lista_ps, r0, rf, len_r, T, N//8, N//16, 'r1')
        # plt.show()

    class grid_varying_mh:
        ...
        # N = 10000
        # E = 12*N
        # lista_ms = np.arange(2,8)
        # T = 1000
        # r0, rf, len_r = 1e-6, 1e+2, 50
        # plot_pcolor_varying_hm_critical_p_rect_grid(N, E, lista_ms, r0, rf, len_r, T, N//8, N//16, 'r1')
        # plt.show()
