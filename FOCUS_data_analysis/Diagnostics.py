# Methods used to analyze data from particle dynamics on analytical MHD equilibria
# Plots the poloidal flux surfaces based on analytical equilibria
# The Diagnostics_dpos (dneg) file contains several properties of the particles (r, theta, z, vr, vth, vz, pitch, E_kev, state, time) at 10 different times 

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import imageio
import matplotlib.colors as colors


plt.rcParams.update({'font.size': 12})
plt.rcParams["mathtext.default"] = 'regular'

# Global Eq Parameters
M = 0.01
R0 = 1
k = 1.43
a = 0.67 / 1.67
Zx = k * a


def Psi(R, Z, params=None):
    # Returns the poloidal flux for a given R and Z, params is a tuple of the coefficients
    if params is None:
        return 0
    S1, S2, C1, C2, C3, C4, C5, C6, C7, C8 = params

    psi2 = R * R
    psi3 = pow(R, 4) - 4 * R * R * Z * Z
    psi4 = R * R * np.log(R) - Z * Z
    psi5 = pow(Z, 4) - 6 * R * R * Z * Z * np.log(R) + 3 * R * R * Z * Z + 3 * pow(R, 4) * np.log(R) / 2 - 15 * pow(R,4) / 8
    psi6 = R * R * pow(Z, 4) - 3 * pow(R, 4) * Z * Z / 2 + pow(R, 6) / 8
    psi7 = pow(Z, 6) + 15 * R * R * pow(Z, 4) * (1 - 2 * np.log(R)) / 2 + 45 * pow(R, 4) * Z * Z * (
            2 * np.log(R) - 2.5) / 4 - 15 * pow(R, 6) * (2 * np.log(R) - 10.0 / 3) / 16
    psi8 = R * R * pow(Z, 6) - 15 * pow(R, 4) * pow(Z, 4) / 4 + 15 * pow(R, 6) * Z * Z / 8 - 5 * pow(R, 8) / 64
    f1 = -(1 + M * M * R * R - np.exp(M * M * R * R)) * S1 / (4 * pow(M, 4))
    f2 = -S2 * Z * Z / 2

    return C1 + C2 * psi2 + C3 * psi3 + C4 * psi4 + C5 * psi5 + C6 * psi6 + C7 * psi7 + C8 * psi8 + f1 + f2


def B(R, Z, params=None):
    # Returns the three components of the magnetic field, params is a tuple of the coefficients
    if params is None:
            return 0
    S1, S2, C1, C2, C3, C4, C5, C6, C7, C8 = params

    Br = -(C3 * (-8 * R * R * Z) + C4 * (-2 * Z) + C5 * (4 * pow(Z, 3) - 12 * R * R * Z * np.log(R) + 6 * R * R * Z)
        + C6 * (4 * R * R * pow(Z, 3) - 3 * pow(R, 4) * Z) + C7 * (45 * pow(R, 4) * Z * (4 * np.log(R) - 5) / 4 + 30 * R * R * pow(Z, 3) * (1 - 2 * np.log(R)) + 6 * pow(Z, 5))
        + C8 * (6 * R * R * pow(Z, 5) - 15 * pow(R, 4) * pow(Z, 3) + 15 * pow(R, 6) * Z / 4) - S2 * Z) / R

    Bt = np.sqrt(1 + 2 * S2 * Psi(R, Z)) / R

    Bz = (C2 * (2 * R) + C3 * (4 * pow(R, 3) - 8 * R * pow(Z, 2)) + C4 * (R + 2 * R * np.log(R)) + C5 * ((6 * (pow(R, 3) - 2 * R * pow(Z, 2)) * np.log(R) - 6 * pow(R, 3)))
        + C6 * (3 * pow(R, 5) / 4 - 6 * pow(R, 3) * pow(Z, 2) + 2 * R * pow(Z, 4)) + C7 * (15 * (9 * pow(R, 5) - 48 * pow(R, 3) * Z * Z - 2 * (3 * pow(R, 5) - 24 * pow(R, 3) * Z * Z + 8 * R * pow(Z, 4)) * np.log(R)) / 8)
        + C8 * (2 * R * pow(Z, 6) - 15 * pow(R, 3) * pow(Z, 4) + 45 * pow(R, 5) * Z * Z / 4 - 5 * pow(R, 7) / 8) - S1 * R * (1 - np.exp(M * M * R * R)) / (2 * M * M)) / R

    return Br, Bt, Bz

def Magnetic_axis_pos(params=None):
    # Returns the position of the magnetic axis, defined as the R in which Psi is maximum
    if params is None:
        return 0
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)
        
    R = np.linspace(R0-a, R0+a, 1000)
    psi = Psi(R, 0, params)
    psi_max = np.max(psi)
    psi_max_pos = R[np.where(psi == psi_max)]
    return psi_max_pos


def Plot_Flux(params, N=400, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, plot_separatrix=False, title="", save=False, reversed=False, show=True):
    # Plots the poloidal flux surfaces based on the analytical equilibria, the coefficients (params) can be given as a tuple or as a text file
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    # Plots the flux surface
    R = np.linspace(Rmin, Rmax, N)
    Z = np.linspace(Zmin, Zmax, N)
    R, Z = np.meshgrid(R, Z)
    if reversed:
        psi = -1*np.where(Psi(R, Z, params)<=0, Psi(R, Z, params), 0)  # We only plot the negative flux surfaces
        plt.contour(R, Z, psi, 10, cmap='plasma')
    else: 
        psi = np.where(Psi(R, Z, params)>=0, Psi(R, Z, params), 0)  # We only plot the postive flux surfaces
        plt.contour(R, Z, psi, 10, cmap='plasma')

    # plt.colorbar()
    if plot_separatrix:
        plt.contour(R, Z, psi, [0], colors='black')
    
    fig = plt.gcf()
    fig.set_figwidth(4.2)
    fig.set_figheight(4.2*np.sqrt(2))

    plt.tick_params(direction='in')
    plt.xlabel(r'$r/R_0$')
    plt.ylabel(r'$z/R_0$')
    plt.xlim(Rmin, Rmax)
    plt.ylim(Zmin, Zmax)
    plt.axis('scaled')


    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png')
    elif show:
        plt.show()

def Plot_Ion_Data(X, Y_, *args, params=None):  # plots the data
    if params is None:
        return 0
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)


    fig, ax = plt.subplots(figsize=(6, 6))

    # Tokamak edge and mag-axis
    ang = np.linspace(0, 2 * np.pi, 1000)
    ax.plot((1 + a) * np.cos(ang), (1 + a) * np.sin(ang), color="black", linewidth=0.4)
    ax.plot((1 - a) * np.cos(ang), (1 - a) * np.sin(ang), color="black", linewidth=0.4)
    ax.plot(Magnetic_axis_pos(params) * np.cos(ang), Magnetic_axis_pos(params) * np.sin(ang), "b--", linewidth=0.4)

    # Data
    ax.scatter(X, Y_, color="red", alpha=0.5, label="Escapadas", s=5)
    k = 0
    while k < len(args):
        ax.scatter(args[k], args[k+1], alpha=0.4, label="")
        k += 2

    print("Data size: ", len(X))

    
    ax.legend(loc='upper left')
    ax.tick_params(direction='in')
    
    ax.set_xlabel(r'$x/R_0$', fontsize=14)
    ax.set_ylabel(r'$y/R_0$', fontsize=14)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.show()

# This function plots a histogram of the poloidal angle of the data, measured from the magnetic axis
def Plot_Poloidal_Angle(rp, zp, rn, zn, dpos_coef, dneg_coef, plot=True, save=False, title="", colors=["red", "blue"]):
    # Magnetics axes position
    m_pos = Magnetic_axis_pos(params=dpos_coef)
    m_neg = Magnetic_axis_pos(params=dneg_coef)
    # Data
    plt.hist(np.arctan2(zp, rp-m_pos), bins=100, density=True, color=colors[0], alpha=0.75, label=r"$\delta>0$")
    plt.hist(np.arctan2(zn, rn-m_neg), bins=100, density=True, color=colors[1], alpha=0.75, label=r"$\delta<0$")
    # r_i-1 because the magnetic axis is at r=1

    plt.legend(loc='best')
    plt.tick_params(direction='in')
    plt.xlabel(r'$\phi$ (ángulo poloidal)', fontsize=14)
    plt.ylabel('Densidad de probabilidad', fontsize=14)
    plt.xlim(-1, np.pi)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=900)
    elif plot:
        plt.show()


# filter an array or series of arrays based on condition
def filter_val(cond, *args):
    c = []
    for arr in args:
        c.append(arr[cond])
    return tuple(c)

# this function plots both energy histograms:
def Plot_Hist_E(A, B=None, label_A="", label_B="", labelx="", bins=40, xlim=100, ylim=None, save=False, title="", legend_title=""):
    # Data
    plt.hist(A, bins=bins, density=True, color="red", alpha=0.5, label=label_A)
    if B is not None:
        plt.hist(B, bins=bins, density=True, color="blue", alpha=0.5, label=label_B)
    # r_i-1 because the magnetic axis is at r=1

    plt.legend(loc='upper left', title=legend_title)

    plt.tick_params(direction='in')
    plt.xlabel(labelx, fontsize=14)
    plt.ylabel('Densidad de probabilidad', fontsize=14)
    plt.xlim(0, xlim)  # keV
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=200)
    plt.show()

    
def E_depositada(E, avg=False, max_E=80): # returns the deposited energy 
    e = np.sum(max_E-E)  # 80 keV is the initial energy of the ions
    if avg:
        e /= len(E)
    return e


def Read_Data(filename, skiprows=0, delimiter='\t', comments='#', unpack=True, dtype=float):
    """
    Reads data from a file and returns the data as a tuple of arrays
    """
    data = np.loadtxt(filename, dtype=dtype, comments=comments, delimiter=delimiter, skiprows=skiprows, unpack=unpack)
    return data


def Plot_stationary_Ion_State(Data, params):
    """
    Plots the stationary ion state in the X-Y plane, taken as the concatenation of the data at different times
    """
    if params is None:
        return 0
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    # Tokamak edge and mag-axis
    fig, ax = plt.subplots()
    ang = np.linspace(0, 2 * np.pi, 1000)
    ax.plot((1 + a) * np.cos(ang), (1 + a) * np.sin(ang), color="black", linewidth=0.4)
    ax.plot((1 - a) * np.cos(ang), (1 - a) * np.sin(ang), color="black", linewidth=0.4)
    ax.plot(Magnetic_axis_pos(params) * np.cos(ang), Magnetic_axis_pos(params) * np.sin(ang), "b--", linewidth=0.4)

    # Data
    #time_it, r, th, state = Data[0], Data[1], Data[2], Data[9]  # position of all particles at all measured times
    Energy, r, th, state = Data[8], Data[1], Data[2], Data[9]  # position of all particles at all measured times
    Energy, r, th, state = filter_val(state != 0, Energy, r, th, state)  # filter out particles that have escaped 
    Energy, r, th, state = filter_val(state != -2, Energy, r, th, state)  # filter out neutral particles
    # Could use np.logical_and ^^

    #ax.scatter(r[:10000] * np.cos(th[:10000]), r[:10000] * np.sin(th[:10000]), c=time_it[:10000], alpha=0.5, label="Estado Estacionario", s=3)
    P = ax.scatter(r * np.cos(th), r * np.sin(th), c=Energy, alpha=0.5, label="", s=3, cmap='plasma')
    # Now lets add a colorbar to the scatter plot:
    cbar = plt.colorbar(P, format='%.0f')
    cbar.set_label(r'$E$ [keV]', fontsize=14)


    print("Data size: ", len(r))
    print("max Energy: ", np.max(Energy), " keV")

    
    ax.legend(loc='upper left')
    ax.tick_params(direction='in')
    
    ax.set_xlabel(r'$x/R_0$', fontsize=14)
    ax.set_ylabel(r'$y/R_0$', fontsize=14)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    #plt.tight_layout()
    plt.show()


def Plot_mean_Energy(time_it_pos, time_it_neg, E_pos, E_neg, save=False, title=""):
    """
    Plots the mean energy of the distribution of ions at different times
    """

    Ep = [np.mean(filter_val(time_it_pos==i, E_pos)) for i in range(10)]
    En = [np.mean(filter_val(time_it_neg==i, E_neg)) for i in range(10)]
    t = [9*i for i in range(10)]

    plt.scatter(t, Ep, color="red", label=r"$\delta>0$", alpha=0.9)
    plt.scatter(t, En, color="blue", label=r"$\delta<0$", alpha=0.9)
    plt.legend(loc='best')
    plt.tick_params(direction='in')
    plt.xlabel(r'$tiempo$ [ms]', fontsize=14)
    plt.ylabel(r'$\overline{E}$ [keV]', fontsize=14)

    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=200)
    plt.show()

def Plot_Energy_pitch_dist(E, pitch, save=False, title="", bins=50):
    """
    Plots a 2d histogram of the energy and pitch angle of the ions
    """
    plt.figure(figsize=(8, 5))
    plt.hist2d(E, pitch, bins=bins, cmap='gnuplot', density=True)
    #plt.clim(0,0.1)
    print("Mean Energy: ", np.mean(E))
    plt.tick_params(direction='in')
    plt.xlabel("Energía [keV]", fontsize=14)
    plt.ylabel('cos(p)', fontsize=14)
    plt.colorbar()
    #plt.tight_layout()
    plt.xlim(0,100)
    plt.ylim(-1,1)
    if save:
        plt.savefig(title + '.png', dpi=200)
    plt.show()


def Plot_Ion_current_dist(r_part, z_part, v_t, params=None, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, M=400, gridlen=70, save=False, title="test"):
    """
    Plots the (toroidal) Ionic current distribution energy distribution based on the toroidal velocity v_t
    """
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.set_figwidth(4.2*np.sqrt(2))
    fig.set_figheight(4.2*np.sqrt(2))

    # We start by plotting the eq separatrix:
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    # Plots the flux surface
    R = np.linspace(Rmin, Rmax, M)
    Z = np.linspace(Zmin, Zmax, M)
    R, Z = np.meshgrid(R, Z)

    psi = np.where(Psi(R, Z, params)>=0, Psi(R, Z, params), 0)  # We only plot the postive flux surfaces
    ax.contour(R, Z, psi, [0], colors='black') # number instead of [0] to plot several countours

    # Lets rearrange Q to a matrix:
    N = gridlen
    # lets create a sqeare grid of points:
    r = np.linspace(Rmin, Rmax, N)
    z = np.linspace(Zmin, Zmax, N)

    # Now lets calculate the deposited energy at each point:
    Q = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            v_t_aux, r_part_aux, z_part_aux = filter_val(r_part>r[i], v_t, r_part, z_part)
            v_t_aux, r_part_aux, z_part_aux = filter_val(r_part_aux<r[(i+1)%N], v_t_aux, r_part_aux, z_part_aux)
            v_t_aux, r_part_aux, z_part_aux = filter_val(z_part_aux>z[j], v_t_aux, r_part_aux, z_part_aux)
            v_t_aux, = filter_val(z_part_aux<z[(j+1)%N], v_t_aux)

            Q[j, i] = np.sum(v_t_aux)
    
    # Now lets plot the deposited energy distribution:
    
    v_min = np.min(Q)
    v_max = np.max(Q)
    Q = np.where(Q>0, Q, 0)  # only for visual effects
    # Otra forma mejor era cambiar todos los 0s de Q por v_min y usar un colormap de blanco a rojo

    norm = colors.TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)

    P = ax.imshow(Q, cmap='seismic', origin='lower', aspect='auto', extent=[R0-a, R0+a, -Zx, Zx], norm=norm)    

    cbar = plt.colorbar(P, format='%.2f')
    cbar.set_label(r'$J$ [$?$]', fontsize=14)
    ax.set_xlabel(r'$r/R_0$', fontsize=14)
    ax.set_ylabel(r'$z/R_0$', fontsize=14)
    plt.xlim(Rmin, Rmax)
    plt.ylim(Zmin, Zmax)
    plt.tick_params(direction='in')
    plt.axis('equal')
    #plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=200)

    plt.show()


if __name__ == "__main__":
    # time taken to run the code
    start_time = time.time()
    
    # Plotting
    coefs_dpos = r"PolFlux_DIIID_cand5\s=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.5676_V=3.4043.txt"
    coefs_dneg = r"PolFlux_DIIID_cand5\s=0.0820_alpha_i=-4.5000_alpha_o=2.0000_A=0.5675_V=3.7152.txt"
    # negI = r"C:\Users\fmart\Desktop\Estudio\Balseiro\Tesis\EqTor_puntosX\PolFlux_DIIID_cand5\negative_Ips=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.3547_V=2.3651.txt"


    
    # Lets read the Diagnostics file using pandas:
    data_p = Read_Data("Diagnostics_dpos.dat")  # Data[i] contains the ith column of the file
    data_n = Read_Data("Diagnostics_dneg.dat")  
    # data contains (time_it, r, theta, z, vr, vth, vz, pitch, E_kev, state, time)

    # print(data_p[0])

    time_it_p, r_p, z_p, state_p, Energy_p, pitch_p, vr_p, vth_p, vz_p = data_p[0], data_p[1], data_p[3], data_p[9], data_p[8], data_p[7], data_p[4], data_p[5], data_p[6] 
    time_it_n, r_n, z_n, state_n, Energy_n, pitch_n, vr_n, vth_n, vz_n = data_n[0], data_n[1], data_n[3], data_n[9], data_n[8], data_n[7], data_n[4], data_n[5], data_n[6]

    vmod_p = np.sqrt(vr_p**2 + vth_p**2 + vz_p**2)
    vmod_n = np.sqrt(vr_n**2 + vth_n**2 + vz_n**2)

    #Plot_stationary_Ion_State(data_p, coefs_dpos)

    time_it_p, r_p, z_p, state_p, Energy_p, pitch_p, vmod_p, vth_p = filter_val(state_p != -2, time_it_p, r_p, z_p, state_p, Energy_p, pitch_p, vmod_p, vth_p)  # filter out neutral particles
    time_it_p, r_p, z_p, state_p, Energy_p, pitch_p, vmod_p, vth_p = filter_val(state_p != 0, time_it_p, r_p, z_p, state_p, Energy_p, pitch_p, vmod_p, vth_p)  # filter out particles that have escaped 

    time_it_n, r_n, z_n, state_n, Energy_n, pitch_n, vmod_n, vth_n = filter_val(state_n != -2, time_it_n, r_n, z_n, state_n, Energy_n, pitch_n, vmod_n, vth_n)  # filter out neutral particles
    time_it_n, r_n, z_n, state_n, Energy_n, pitch_n, vmod_n, vth_n = filter_val(state_n != 0, time_it_n, r_n, z_n, state_n, Energy_n, pitch_n, vmod_n, vth_n)  # filter out particles that have escaped
    
    time_it_n, Energy_n, pitch_n, vmod_n, vth_n, r_n, z_n = filter_val(Energy_n >= 0, time_it_n, Energy_n, pitch_n, vmod_n, vth_n, r_n, z_n)
    time_it_p, Energy_p, pitch_p, vmod_p, vth_p, r_p, z_p = filter_val(Energy_p >= 0, time_it_p, Energy_p, pitch_p, vmod_p, vth_p, r_p, z_p)

    time_it_n, Energy_n, pitch_n, vmod_n, state_n, vth_n, r_n, z_n = filter_val(time_it_n > 0, time_it_n, Energy_n, pitch_n, vmod_n, state_n, vth_n, r_n, z_n)
    time_it_p, Energy_p, pitch_p, vmod_p, state_p, vth_p, r_p, z_p = filter_val(time_it_p > 0, time_it_p, Energy_p, pitch_p, vmod_p, state_p, vth_p, r_p, z_p)

    pitch_n = pitch_n/vmod_n  # could actually do this from the start, before filtering
    pitch_p = pitch_p/vmod_p

    Plot_Ion_current_dist(r_n, z_n, vth_n, coefs_dneg, save=False, title="Ion_current_dist_dpos")

    #Plot_Energy_pitch_dist(filter_val(time_it_p==0, Energy_p)[0], filter_val(time_it_p==0, pitch_p)[0], bins=100, save=True, title="Energy_pitch_dist_dpos")
    Plot_Energy_pitch_dist(filter_val(time_it_n<5, Energy_n)[0], filter_val(time_it_n<5, pitch_n)[0], bins=100, save=False, title="Energy_pitch_dist_dneg")

    # Distribución de frenamiento
    Plot_Hist_E(filter_val(time_it_p==9, (Energy_p)), filter_val(time_it_n==9, (Energy_n)), r"$\delta > 0$", r"$\delta < 0$", xlim=90, labelx="E [kev]")  
    
    #Plot_Hist_E(filter_val(time_it_p<10, (pitch_p)), filter_val(time_it_n<10, (pitch_n)), r"$\delta > 0$", r"$\delta < 0$", xlim=90, labelx="E [kev]")  

    # distribución de velocidad normalizada:
    #Plot_Hist_E(filter_val(time_it_p<5, np.sqrt(Energy_p/80)), filter_val(time_it_n<5, np.sqrt(Energy_n/80)), r"$\delta > 0$", r"$\delta < 0$",xlim=1, bins=60,labelx=r"$v/v_0$", save=True, title="v_norm_hist_t=36ms")

    # Mean Energy
    #Plot_mean_Energy(time_it_p, time_it_n, Energy_p, Energy_n, save=False, title="mean_Energy")

    print(f"Max E por dpos: {np.max(Energy_p)}\nMax E por dneg: {np.max(Energy_n)}")
    #time_it_p = filter_val(Energy_p==np.max(Energy_p), time_it_p)
    #print(time_it_p[0])
    #print(time_it_p)
    
    # analyze escaped and confined particles:
    #last_states_p = filter_val(time_it_p==9, state_p)
    #last_states_n = filter_val(time_it_n==9, state_n)  # Recordar que filter_val devuelve tuplas!!!

    """
    part_num = 200000
    banana_p = []
    banana_n = []
    for i in range(10):
        last_states_p= filter_val(time_it_p==i, state_p)
        last_states_n= filter_val(time_it_n==i, state_n)

        banana_p.append(np.sum(np.where(last_states_p[0]==1, 1, 0)))
        banana_n.append(np.sum(np.where(last_states_n[0]==1, 1, 0)))
    print(f"Banana Orbits, dpos: {banana_p}\n dneg {banana_n}")
    """
    
    #print(f"banana particles dpos: {np.sum(np.where(last_states_p[0]==1, 1, 0))}\nbanana particles dneg: {np.sum(np.where(last_states_n[0]==1, 1, 0))}")
    #print(f"confined particles dpos: {np.sum(np.where(last_states_p[0]==2, 1, 0))}\nconfined particles dneg: {np.sum(np.where(last_states_p[0]==2, 1, 0))}")


    # Create energy hist GIF
    """
    images = []
    for i in range(1,10):  # plots Energy histogram of particles at time > 0. At time = 0 we have a delta function @ 80keV
        Ep, p_p = filter_val(time_it_p<=i, (Energy_p), (pitch_p))
        En, p_n = filter_val(time_it_n<=i, (Energy_n), (pitch_n))
        #v_p = np.sqrt(Ep/80)
        #v_n = np.sqrt(En/80)

        Plot_Energy_pitch_dist(Ep, p_p, bins=100, save=True, title="fig"+str(i))


        #Plot_Hist_E(v_p, v_n, r"$\delta > 0, \overline{v/v_0}=$"+f"{np.mean(v_p):.2f}", r"$\delta < 0, \overline{v/v_0}=$"+f"{np.mean(v_n):.2f}", labelx=r"$v/v_0$",
        #save=True, title="fig"+str(i), legend_title=f"t={9*i} ms", ylim=7.5, xlim=1)

        #Plot_Hist_E(Ep, En, r"$\delta > 0, \overline{E}=$"+f"{np.mean(Ep):.2f}", r"$\delta < 0, \overline{E}=$"+f"{np.mean(En):.2f}", labelx="E [kev]",
        #save=True, title="fig"+str(i), legend_title=f"t={9*i} ms", ylim=1, xlim=90)
        c = imageio.imread("fig"+str(i)+".png")
        images.append(c)
    images.append(images[-1])  # so that the last image lasts twice as long
    imageio.mimsave("Epitch_distribution_over_time.gif", images, duration=0.7)
    """


    # Lets proceed to plot the poloidal coordinates of the particles in the equilibria
    """
    Plot_Flux(coefs_dpos, show=False)
    plt.scatter(r_p, z_p, c=time_it_p, s=12, label="", cmap="viridis", alpha=0.3)
    plt.legend(loc='lower right')
    plt.show()

    Plot_Flux(coefs_dneg, show=False)
    plt.scatter(r_n, z_n, c=time_it_n, s=12, label="", cmap='viridis', alpha=0.3)
    plt.legend(loc='lower right')
    plt.show()
    """
    # Create positions GIF
    """
    images = []
    for i in range(10):
        Plot_Flux(coefs_dneg, show=False)
        R_n = filter_val(time_it_n==i, r_n)
        Z_n = filter_val(time_it_n==i, z_n)
        plt.scatter(R_n, Z_n, c='black', s=12, label=f"t={9*i} ms", alpha=0.8)
        plt.legend(loc='best')
        plt.savefig("test_stat_state/fig"+str(i)+".png")
        plt.show()
        c = imageio.imread("test_stat_state/fig"+str(i)+".png")
        images.append(c)  
    imageio.mimsave("test_stat_state/movie.gif", images, duration=0.6)
    """


    # time taken to run the code
    print("--- %s seconds ---" % (time.time() - start_time))
