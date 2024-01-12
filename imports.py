import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit
import matplotlib as mpl
from scipy.optimize import curve_fit

@jit(nopython=True)
def mean(lista):
    media = 0
    for el in lista:
        media += el/len(lista)
    return media

@jit(nopython=True)
def matrix_mean(matrix):
    """
    Computes the mean of a matrix over the zero axis.
    """
    mean = np.zeros(matrix.shape[1:])
    rol_number = matrix.shape[0]
    for i in range(rol_number):
        mean += matrix[i,:]/rol_number
    return mean

@jit(nopython=True)
def matrix_std(matrix, mean):
    """
    Computes the std of a matrix over the zero axis.
    """
    var = np.zeros(matrix.shape[1:])
    rol_number = matrix.shape[0]
    for j in range(rol_number):
        var += (matrix[j,:] - mean)**2 /rol_number
    std = np.sqrt(var)
    return std

@jit(nopython=True)
def adjacency_maker(N,E,m,h):
    """Função que aleatoriza e determina a matriz de adjacência NxN. Ela explicita qual neurônio j da coluna se liga com i da linha, nessa ordem. Ou seja, j -> i.

    Args:
        N (int): Tamanho da rede.
        E (int): Quantidade de arestats total da rede.
        m (int): Quantidade de submódulos.
        h (int): Quantidade de hierarquias.
    Returns:
        NxN float64: Matriz de adjacência, de 0s ou ps.
    """
    def sort_ci(N_i, p_i):
        c_i = np.zeros((N_i,N_i))
        for i in range(N_i):
            for j in range(N_i):
                if i != j:
                    c_i[i,j] = 1 if np.random.uniform(0,1) < p_i else 0
        return c_i
    
    E_i = E/(h+1)
    CIJ = np.zeros((N,N))

    for i in range(h):
        A_i = (m-1)/(m**(i+1))
        p_i = E_i/(A_i*(N**2))
        N_i = int(N*(1/m)**i)
        N_c = int(N/N_i)
        
        for j in range(N_c):
            c_i = sort_ci(N_i, p_i)
            r_0 = int(1 + j*N_i)
            r_1 = int(r_0 + N_i - 1)
            CIJ[(r_0-1):(r_1) , (r_0-1):(r_1)] = c_i

    return CIJ

def CIJ_to_coon(adjac):
    N = len(adjac)
    coon = []
    for i in range(N):
        for j in range(N):
            if adjac[i,j] == 1:
                coon.append((i,j))
    return coon

@jit(nopython=True)
def CIJ_to_P(CIJ):
    N = len(CIJ)
    max_K = 0
    for i in range(N):
        K = np.count_nonzero(CIJ[i,:])
        max_K = K if K > max_K else max_K
    P = np.zeros((N,max_K))
    for i in range(N):
        k=0
        for j in range(N):
            if CIJ[i,j] != 0:
                p = CIJ[i,j]
                P[i,k] = j
                k+=1
        P[i,k:N].fill(-1)
    return P

@jit(nopython=True)
def P_to_CIJ(P):
    N, max_k = P.shape
    CIJ = np.zeros((N,N))
    for i in range(N):
        for j in range(max_k):
            if P[i,j] != -1:
                CIJ[i,int(P[i,j])] = 1
            else:
                break
    return CIJ

def Adjac_file_man(N, E, m, h, mode):
    """Função que gerencia as matrizes de vizinhos N x max_K, construindo, lendo, escrevendo num arquivo quando necessário, e retornando-a. Ela explicita qual neurônio da entrada da matriz se liga com a linha i. Ou seja, i <- P[i, k].

    Args:
        N (int): Tamanho da rede.
        E (int): Quantidade de arestats total da rede.
        m (int): Quantidade de submódulos.
        h (int): Quantidade de hierarquias.
        mode (string): Modo de abertura ou escrita. ```w``` se quiser escrever uma nova ou ```r{net}``` para ler alguma ```net``` específica.
    Returns:
        Nxmax_k float64: Matriz de vizinhos.
    """
    folder_adress = f"MHN/Data/Networks/N={N}/m={m}_h={h}"
    
    folder_existance_check(folder_adress)
    
    datas = os.listdir(folder_adress)
    adresses_that_fit = []
    for arq in datas:
        qtos_ = 0
        for i, carac in enumerate(arq):
            if carac=="_":
                qtos_ += 1
            if qtos_ == 3:
                pos_ = i
                break
        if arq[:pos_] == f"K={E//N}_m={m}_h={h}":
            adresses_that_fit.append(folder_adress + "/" + arq)
    
    iter = len(adresses_that_fit) + 1
    common_adress = f"MHN/Data/Networks/N={N}/m={m}_h={h}/"
    folder_existance_check(common_adress)
        
    if mode[0] == "w":
        A = adjacency_maker(N, E, m, h)
        P = CIJ_to_P(A)
        np.savetxt(common_adress+f"K={E//N}_m={m}_h={h}_{iter}.txt", P, "%i", delimiter="\t", newline="\n")
        oi = open(common_adress+f"K={E//N}_m={m}_h={h}_{iter}.txt", "a")
        oi.write(f"#{1/autoval_JG(A)}")
        oi.close()
        p_crit = 1/autoval_JG(A)
        net = iter
    elif mode[0] == "r":
        P = np.loadtxt(common_adress+f"K={E//N}_m={m}_h={h}_{mode[1:]}.txt",dtype="int",delimiter="\t", comments="#")
        oi = open(common_adress+f"K={E//N}_m={m}_h={h}_{mode[1:]}.txt", "r")
        for l in range(N):
            oi.readline()
        oi.seek(oi.tell() + 1)
        p_crit = float(oi.readline())
        oi.close()
        net = mode[1:]
    return P, p_crit, net
    
@jit(nopython=True)    
def autoval_JG(matrix):
    """Calculates the largest eigenvalue of a matrix by the iterative JG method.

    Args:
        matrix (float64): Desired square matrix

    Returns:
        float: The eigenvalue.
    """
    iterations=100
    b = np.random.rand(len(matrix))
    for i in range(iterations):
        prox_b = (matrix @ b)/np.linalg.norm(matrix @ b)
        mi = (prox_b.reshape(1,-1)) @ (matrix @ b)
        b = prox_b
    # Converter para float por causa do numba
    for mi_temp in mi:
        final_mi = mi_temp
    return final_mi

@jit(nopython=True)
def iterator_count(CIJ, x, states):
    """Iterates one timestep of a network, according to the K&C 2006 model.
    Deprecated, because ```CIJ``` is too memory and time-consuming for a typical iteration. Use ```P_iterator``` instead.

    Args:
        CIJ ((N,N) array): The connectivity matrix, of p"s and 0"s.
        x ((N,) array): Current state of the network.
        states (int): Number of possible states. 0 and 1 are reserved for the inactive and polarized state, so ```states > 2```.

    Returns:
        (N,) array: Next state of the system.
        int: The total number of excitations.
    """
    N = len(x)
    x_new = np.copy(x)
    n_est = 0
    
    for j in range(N):
        if x[j] != 0:
            x_new[j] = (x[j] + 1)%states
        else:
            for i in range(N):
                con = CIJ[i,j]
                u = np.random.uniform(0,1)
                if x[i] == 1 and u<con:
                    x_new[j] = 1
                    n_est += 1
                    break
    return x_new, n_est

@jit(nopython=True)
def P_iterator_count(P, x, states, p):
    """Iterates one timestep of a network, according to the K&C 2006 model.
    Updated and recommended iteration step. See ```iterator``` for a legacy model, that uses ```CIJ``` instead of ```P```.

    Args:
        P ((N,k) array): The neighborhood matrix, with the value on the i-th row indicating the index of a connection from the element to i.
        x ((N,) array): Current state of the network.
        states (int): Number of possible states. 0 and 1 are reserved for the inactive and polarized state, so ```states > 2```.
        p (float): Real probability of a connection between the nodes.

    Returns:
        (N,) array: Next state of the system.
        int: The total number of excitations.
    """
    N, k = P.shape
    
    x_new = np.copy(x)
    n_est = 0
    for i in range(N):
        if x[i] != 0:
            x_new[i] = (x[i]+1)%states # Caminha para frente. 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 0.
        else: # Caso o neurônio não esteja polarizado
            for j in range(k): # Começa a caminhar pela matriz de vizinhanças
                con = int(P[i,j])
                if con != -1:
                    u = np.random.uniform(0,1)
                    if x[con] == 1 and u<p: # Se o neurônio tiver uma conexão polarizado
                        x_new[i] = 1
                        n_est += 1
                        break
                else:
                    break
    return x_new, n_est

@jit(nopython=True)
def P_iterator_external_count(P, x, states, p, r):
    """Iterates one timestep of a network, according to the K&C 2006 model.
    Updated and recommended iteration step. See ```iterator``` for a legacy model, that uses ```CIJ``` instead of ```P```.

    Args:
        P ((N,k) array): The neighborhood matrix, with the value on the i-th row indicating the index of a connection from the element to i.
        x ((N,) array): Current state of the network.
        states (int): Number of possible states. 0 and 1 are reserved for the inactive and polarized state, so ```states > 2```.
        p (float): Real probability of a connection between the nodes.

    Returns:
        (N,) array: Next state of the system.
    """
    N, k = P.shape
    
    x_new = np.copy(x)
    n_est = 0
    for i in range(N):
        if x[i] != 0:
            x_new[i] = (x[i]+1)%states # Caminha para frente. 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 0.
        else: # Caso o neurônio não esteja polarizado
            for j in range(k): # Começa a caminhar pela matriz de vizinhanças
                con = int(P[i,j])
                if con != -1:
                    u = np.random.uniform(0,1)
                    if x[con] == 1 and u<p: # Se o neurônio tiver uma conexão polarizado
                        x_new[i] = 1
                        n_est += 1
                        break
                else:
                    break
            est = np.random.uniform(0,1)
            if est <= 1-np.exp(-r*1):
                x_new[i] = 1
    return x_new, n_est

@jit(nopython=True)
def initial_position(N, where_nodes_init, qtde_nodes_init):
    """Randomizes an initial condition for the network state.

    Args:
        N (int): Size of the network.
        where_nodes_init (int): The first nodes to be activated or not.
        qtde_nodes_init (int): How many nodes will be activated

    Returns:
        (N,) float64: The initial array.
    """
    x_inic = np.zeros(N)
    raffled_nodes = np.random.randint(0, where_nodes_init, qtde_nodes_init)
    for i in raffled_nodes:
        x_inic[i] = 1
    return x_inic

@jit(nopython=True)
def time_evaluation(CIJ, states, T, wni, qni):
    """Iterates over time the states of a network.
    Deprecated, because ```CIJ``` is too memory and time-consuming for a typical iteration. Use ```P_iterator``` instead.
    
    Args:
        CIJ ((N,N) array): The connectivity matrix, of p"s and 0"s.
        states (int): Number of possible states. 0 and 1 are reserved for the inactive and polarized state, so ```states > 2```.
        T (int): Number of Time steps to be evaluated.
        wni (int): The first ```wni``` nodes that will (probably) be activated. See ```initial_position``` for details.
        qni (int): Quantity of initially active nodes, of ```wni``` possible ones.

    Returns:
        (N,) array: The final state of the system after ```T``` 
        (T,) array: List with all the density of active nodes over time.
    """
    N = len(CIJ)
    rho = np.zeros(T)
    at_excit_count = 0
    
    for i in range(T):
        if i == 0:
            x = initial_position(N, wni, qni)
            ot_excit_count = qni
        else:
            x, ot_excit_count = iterator_count(CIJ, x, states)
        rho[i] = list(x).count(1)/N
        at_excit_count += ot_excit_count
    return x, rho

@jit(nopython=True)
def P_time_evaluation(P, p, states, T, wni, qni):
    """Iterates over time the states of a network.
    Updated and recommended iteration step. See ```iterator``` for a legacy model, that uses ```CIJ``` instead of ```P```.

    Args:
        P ((N,k) array): The neighborhood matrix, with the value on the i-th row indicating the index of a connection from the element to i.
        states (int): Number of possible states. 0 and 1 are reserved for the inactive and polarized state, so ```states > 2```.
        T (int): Number of Time steps to be evaluated.
        wni (int): The first ```wni``` nodes that will (probably) be activated. See ```initial_position``` for details.
        qni (int): Quantity of initially active nodes, of ```wni``` possible ones.

    Returns:
        (N,) array: The final state of the system after ```T``` iterations.
        (T,) array: List with all the density of active nodes over time.
    """
    N, k = P.shape
    rho = np.zeros(T)
    at_excit_count = 0
    
    for i in range(T):
        if i == 0:
            x = initial_position(N, wni, qni)
            ot_excit_count = qni
        else:
            x, ot_excit_count = P_iterator_count(P, x, states, p)
        if np.sum(x) == 0:
            return x, rho
        rho[i] = list(x).count(1)/N
        at_excit_count += ot_excit_count
        
    return x, rho

@jit(nopython=True)
def P_time_evaluation_external(P, p, states, T, wni, qni, r):
    """Iterates over time the states of a network.
    Updated and recommended iteration step. See ```iterator``` for a legacy model, that uses ```CIJ``` instead of ```P```.

    Args:
        P ((N,k) array): The neighborhood matrix, with the value on the i-th row indicating the index of a connection from the element to i.
        states (int): Number of possible states. 0 and 1 are reserved for the inactive and polarized state, so ```states > 2```.
        T (int): Number of Time steps to be evaluated.
        wni (int): The first ```wni``` nodes that will (probably) be activated. See ```initial_position``` for details.
        qni (int): Quantity of initially active nodes, of ```wni``` possible ones.

    Returns:
        (N,) array: The final state of the system after ```T``` iterations.
        (T,) array: List with all the density of active nodes over time.
    """
    N, k = P.shape
    rho = np.zeros(T)
    at_excit_count = 0
    
    for i in range(T):
        if i == 0:
            x = initial_position(N, wni, qni)
            ot_excit_count = qni
        else:
            x, ot_excit_count = P_iterator_external_count(P, x, states, p, r)
        rho[i] = list(x).count(1)/N
        at_excit_count += ot_excit_count
        
    return x, rho

@jit(nopython=True)
def degree_calc(P):
    N, max_k = P.shape
    K = 0
    for i in range(N):
        K += list(P[i,:]).count(-1)
        K -= max_k
    return -K/N

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

def find_F_0(lista_rs, lista_Fs, lim):
    rs_adjust = lista_rs[lista_rs < lim]
    Fs_adjust = lista_Fs[:len(rs_adjust)]
    def poly(x, a, b, c):
        return a*x** b + c
    param, _ = curve_fit(poly, rs_adjust, Fs_adjust, maxfev=1500)
    return param

def leitura_ate_tab(linha):
    el = []
    antigo_barrat = 0
    for i in range(1,len(linha)):
        if linha[i] == "\t":
            barrat = i
            if antigo_barrat != 0:
                el.append(linha[(antigo_barrat+1):barrat])
            else:
                el.append(linha[(antigo_barrat):barrat])
            antigo_barrat = barrat
    el.append(linha[(barrat+1):-1])
    return el

def folder_existance_check(file_adress):
    if file_adress[-1] != "/":
        file_adress += "/"
    if os.path.exists(file_adress):
        return
    bars = [0]
    for i, carac in enumerate(file_adress):
        if carac == "/":
            bars.append(i)
    for j in range(1,len(bars)):
        if os.path.exists(file_adress[:bars[j]]) == False:
            directory = str(file_adress[(bars[j-1]+1):bars[j]])
            parent_dir = str(file_adress[:bars[j-1]])
            path = os.path.join(parent_dir,directory)
            os.mkdir(path)
