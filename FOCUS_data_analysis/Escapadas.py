# Plots the poloidal flux surfaces based on analytical equilibria
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.misc import derivative
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

def Plot_B(params=None, N=400, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx):
    # Plots the magnetic field components
    if params is None:
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    R = np.linspace(Rmin, Rmax, N)
    Z = np.linspace(Zmin, Zmax, N)
    Br, Bt, Bz = B(R, Z, params)
    plt.figure()
    plt.plot(R, Br, label='Br')
    plt.plot(R, Bt, label='Bt')
    plt.plot(R, Bz, label='Bz')

    # lets plot the divergence of the field
    """
    dBr = np.gradient(R*Br, R)/R
    dBz = np.gradient(Bz, Z)
    divB = dBr + dBz
    plt.plot(R, divB, label='divB')

    plt.legend()"""
    plt.xlabel('R')
    plt.ylabel('B')
    plt.show()

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
def Plot_Poloidal_Angle(rp, zp, rn, zn, dpos_coef, dneg_coef, Xpoints=None, plot=True, save=False, title="", colors=["red", "blue"]):
    # Magnetics axes position
    m_pos = Magnetic_axis_pos(params=dpos_coef)
    m_neg = Magnetic_axis_pos(params=dneg_coef)

    if Xpoints is not None:
        # X points poloidal angles:
        Xpoints_pos = np.arctan2(Xpoints[0][1], Xpoints[0][0]-m_pos)
        Xpoints_neg = np.arctan2(Xpoints[1][1], Xpoints[1][0]-m_neg)

        # plot the x points
        plt.axvline(x=Xpoints_pos, ymin=0, ymax=1, color='red', linestyle='--', linewidth=1.2, label=r"Divertor $\delta>0$")
        plt.axvline(x=Xpoints_neg, ymin=0, ymax=1, color='blue', linestyle='--', linewidth=1.2, label=r"Divertor $\delta<0$")

    # Data 
    plt.hist(np.arctan2(zp, rp-m_pos), bins=100, density=True, color=colors[0], alpha=0.5, label=r"$\delta>0$")
    plt.hist(np.arctan2(zn, rn-m_neg), bins=100, density=True, color=colors[1], alpha=0.5, label=r"$\delta<0$")
    # r_i-1 because the magnetic axis is at r=1

    plt.legend(loc='best')
    plt.tick_params(direction='in')
    plt.xlabel(r'$\phi$', fontsize=14)
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
def Plot_Hist_E(A, B, label_A="", label_B="", labelx="", bins=40, save=False, title=""):
    # Data
    plt.hist(A, bins=bins, density=True, color="red", alpha=0.5, label=label_A)
    plt.hist(B, bins=bins, density=True, color="blue", alpha=0.5, label=label_B)
    # r_i-1 because the magnetic axis is at r=1

    plt.legend(loc='best')
    plt.tick_params(direction='in')
    plt.xlabel(labelx, fontsize=14)
    plt.ylabel('Densidad de probabilidad', fontsize=14)
    plt.xlim(0, 90)  # keV
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=900)
    plt.show()

    
def E_depositada(E, avg=False, max_E=80): # returns the deposited energy 
    e = np.sum(max_E-E)  # 80 keV is the initial energy of the ions
    if avg:
        e /= len(E)
    return e

def Plot_lost_over_time(times_of_loss, color="red", label="", show=False, save=False, title="", particles=200000):
    # Lets plot the particles that escaped over time
    Time = np.linspace(0, np.max(times_of_loss), 10000)

    N = [np.sum(times_of_loss<t) for t in Time]
    Time = Time/96000  # the 96000 converts 0.16*cyclotron periods to ms
    N = np.array(N)/particles

    plt.plot(Time[1:], N[1:], color=color, label=label)
    #plt.tight_layout()
    plt.legend(loc='best')
    plt.tick_params(direction='in')
    plt.xlabel("Tiempo [ms]", fontsize=14)
    plt.ylabel('Proporción de perdidas', fontsize=14)
    
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=900)
    if show:
        plt.show()


if __name__ == "__main__":
    # time taken to run the code
    start_time = time.time()
    
    # Plotting
    coefs_dpos = r"PolFlux_DIIID_cand5\s=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.5676_V=3.4043.txt"
    coefs_dneg = r"PolFlux_DIIID_cand5\s=0.0820_alpha_i=-4.5000_alpha_o=2.0000_A=0.5675_V=3.7152.txt"
    # negI = r"C:\Users\fmart\Desktop\Estudio\Balseiro\Tesis\EqTor_puntosX\PolFlux_DIIID_cand5\negative_Ips=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.3547_V=2.3651.txt"


    # Plot the escaped particles
    Escaped_dpos_Filename = "escapadas_dpos.dat"
    Escaped_dneg_Filename = "escapadas_dneg.dat"

    r_p, theta_p, z_p, E_p, t_p = np.loadtxt(Escaped_dpos_Filename, float, comments='#', delimiter='\t', skiprows=0, unpack=True)
    r_n, theta_n, z_n, E_n, t_n = np.loadtxt(Escaped_dneg_Filename, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    Plot_lost_over_time(t_p, color="red", label=r"$\delta>0$", show=False, save=False, title="Escaped_particles_over_time")
    Plot_lost_over_time(t_n, particles=100000, color="blue", label=r"$\delta<0$", show=False, save=False, title="Escaped_particles_over_time")
    plt.show()


    # lets filter the particles escaped at early times (tn<1600=0.02ms)
    time_delimiter=8000   # 8000 -> ~0.1ms, enough for a particle to complete a single orbit  
    t_p_filt, r_p_filt, theta_p_filt, z_p_filt, E_p_filt = filter_val(t_p>time_delimiter, t_p, r_p, theta_p, z_p, E_p)
    t_n_filt, r_n_filt, theta_n_filt, z_n_filt, E_n_filt = filter_val(t_n>time_delimiter, t_n, r_n, theta_n, z_n, E_n)
    #t_p_filt, r_p_filt, theta_p_filt, z_p_filt, E_p_filt = filter_val(t_p_filt<3800000*1.5, t_p_filt, r_p_filt, theta_p_filt, z_p_filt, E_p_filt)
    #t_n_filt, r_n_filt, theta_n_filt, z_n_filt, E_n_filt = filter_val(t_n_filt<3800000*1.5, t_n_filt, r_n_filt, theta_n_filt, z_n_filt, E_n_filt)
    #print(t_p_filt)
    #print(t_n_filt)

    # Plot positive triangularity:
    
    #Plot_Flux(coefs_dpos, reversed=False, show=False)    

    """
    File1 = "banana_ejemplo_dpos.txt"
    File2 = "pasante_ejemplo_dpos.txt"
    r_ = np.loadtxt(File1, float, comments='#', delimiter='\t', skiprows=0, usecols=0)
    Z_ = np.loadtxt(File1, float, comments='#', delimiter='\t', skiprows=0, usecols=2)

    plt.plot(r_[:18000]-0.1, Z_[:18000], color="blue", linewidth=0.3, label="Órbita atrapada")

    r_ = np.loadtxt(File2, float, comments='#', delimiter='\t', skiprows=0, usecols=0)
    Z_ = np.loadtxt(File2, float, comments='#', delimiter='\t', skiprows=0, usecols=2)

    plt.plot(r_[:8000], Z_[:8000], color="red", linewidth=0.3, label="Órbita pasante")

    """
    #plt.scatter(r_p_filt, z_p_filt, c='black', s=12, label="Escapadas")
    #plt.legend(loc='lower right')
    #plt.savefig("dpos_escapadas.png", dpi=200)
    #Rx = R0-0.61*a
    #plt.plot((Rx, R0+a), (0, Zx), 'k-', linewidth=1)
    #plt.show()
    
    # Plot negative triangularity:
    """
    File1 = "Trajectories_test_000.txt"
    File2 = "pasante_ejemplo_dneg.txt"
    r_ = np.loadtxt(File1, float, comments='#', delimiter='\t', skiprows=0, usecols=0)
    Z_ = np.loadtxt(File1, float, comments='#', delimiter='\t', skiprows=0, usecols=2)

    plt.plot(r_[:42500], Z_[:42500], color="blue", linewidth=0.3, label="Órbita atrapada")

    r_ = np.loadtxt(File2, float, comments='#', delimiter='\t', skiprows=0, usecols=0)
    Z_ = np.loadtxt(File2, float, comments='#', delimiter='\t', skiprows=0, usecols=2)

    plt.plot(r_[:22200], Z_[:22200], color="red", linewidth=0.3, label="Órbita pasante")
    """
    #Plot_Flux(coefs_dneg, reversed=False, show=False)
    #Rx = R0+0.61*a
    #print(Rx, Zx, R0-a)
    #plt.plot((Rx, R0-a), (0, Zx), 'k-', linewidth=1)
    #plt.scatter(r_n_filt, z_n_filt, c='black', s=12, label="Escapadas")
    #plt.legend(loc='lower left')
    #plt.savefig("dneg_escapadas.png", dpi=200)
    #plt.show()
    
    
    # Plot_B(coefs_dpos)

    #print("Magnetic axis position for dpos: ", Magnetic_axis_pos(coefs_dpos),"\nMagnetic axis position for dneg: ", Magnetic_axis_pos(coefs_dneg))
    
    print(f"Data size dpos: {len(E_p)}\n Filtered data size dpos: {len(E_p_filt)}")
    print(f"Data size dneg: {len(E_n)}\n Filtered data size dneg: {len(E_n_filt)}")
    

    #plt.hist(E_n_filt, bins=50)
    #Plot_Hist_E(E_p_filt, E_n_filt, r"$\delta>0$", r"$\delta<0$", "E [keV]", bins=20)
    #plt.hist(t_p_filt/0.16, bins=100)

    #plt.hist(E_n, bins=50)

    E_depositada_p = E_depositada(E_p_filt, avg=True)
    E_depositada_n = E_depositada(E_n_filt, avg=True)
    print(f"Cociente de E depositada: {E_depositada_n/E_depositada_p}")
    print(f"{E_depositada_p=}")
    print(f"{E_depositada_n=}")
    print(f"Max E_p: {max(E_p_filt)}\nMax E_n: {max(E_n_filt)}")
    


    #Plot_Ion_Data(r_p_filt * np.cos(theta_p_filt), r_p_filt * np.sin(theta_p_filt), params=coefs_dpos)
    #Plot_Ion_Data(r_n_filt * np.cos(theta_n_filt), r_n_filt * np.sin(theta_n_filt), params=coefs_dneg)

    Plot_Poloidal_Angle(r_p_filt, z_p_filt, r_n_filt, z_n_filt, coefs_dpos, coefs_dneg, Xpoints=[[0.755, 0.5737],[1.2447, 0.5737]], plot=False, colors=["red", "blue"])
    #Plot_Poloidal_Angle(r_p, z_p, r_n, z_n, coefs_dpos, coefs_dneg, plot=False, colors=["orange", "blueviolet"])
    #plt.savefig("hist_escapadas.png", dpi=200)
    plt.show()

    # time taken to run the code
    print("--- %s seconds ---" % (time.time() - start_time))
