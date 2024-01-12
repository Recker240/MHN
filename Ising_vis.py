import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from numba import jit

J, kb = 1, 1

def IniSpins():
    s = np.random.choice([-1,1], (L,L), p=[0.4,0.6])
    ezinho=0
    for k in range(L):
        for l in range(L):
            vizs = s[(k-1)%L, l] + s[(k+1)%L, l] + s[k, (l+1)%L] + s[k, (l-1)%L]
            ezinho += -J*s[k, l]*vizs
    return s, ezinho

@jit(nopython=True)
def Delta_E(s, i_s, j_s):
    vizs = s[(i_s-1)%L, j_s] + s[(i_s+1)%L, j_s] + s[i_s, (j_s+1)%L] + s[i_s, (j_s-1)%L]
    var = 2*J*s[i_s, j_s]*vizs
    return var

@jit(nopython=True)
def MCStep(s, T, E):
    N = L**2
    for k in range(N):
        i_sor = np.random.randint(0,L)
        j_sor = np.random.randint(0,L)
        var_E = Delta_E(s, i_sor, j_sor)
        s[i_sor,j_sor] *= -1
        if var_E>0:
            u = np.random.rand()
            if u > np.exp(-var_E/(kb*T)):
                s[i_sor, j_sor] = s[i_sor, j_sor]*-1
                var_E = 0
        E += var_E
    return s, E

if __name__ == "__main__":
    NMC = 1000
    L = 160
    T1, T2 = 1.3, 2.5
    
    E1 = np.zeros(NMC)
    E2 = np.zeros(NMC)
    s1, E1[0] = IniSpins()
    s2, E2[0] = np.copy(s1), E1[0]
    y, x = np.meshgrid(np.arange(L), np.arange(L))
    
    fig, ax = plt.subplots(ncols=2, figsize=(7,3))
    est1 = ax[0].pcolormesh(x, y, s1, cmap="cool")
    est2 = ax[1].pcolormesh(x, y, s2, cmap="cool")
    fig.tight_layout()
    plt.pause(2)

    def init():
        est1.set_array([])
        est2.set_array([])
        return est1, est2
    
    def animate(n):
        si1, E1[n] = MCStep(s1, T1, E1[n-1])
        est1.set_array(si1.ravel())

        si2, E2[n] = MCStep(s2, T2, E2[n-1])
        est2.set_array(si2.ravel())
        
        return est1, est2
    
    
    ani = FuncAnimation(fig, animate, frames=NMC, interval = 30)
    ani.save(f"IC/MHN/Figs/Ising_vis.mp4",dpi=300)
    # plt.show()