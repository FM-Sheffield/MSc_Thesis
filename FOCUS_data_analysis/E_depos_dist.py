# Methods used to analyze data from particle dynamics on analytical MHD equilibria
# Plots the poloidal flux surfaces based on analytical equilibria
# The Diagnostics_dpos (dneg) file contains several properties of the particles (r, theta, z, vr, vth, vz, pitch, E_kev, state, time) at 10 different times 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import pandas as pd
import imageio

plt.rcParams.update({'font.size': 12})
plt.rcParams["mathtext.default"] = 'regular'

# Global Eq Parameters
M = 0.01
R0 = 1
k = 1.43
delta = 0.61
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

    plt.legend(loc='best', title=legend_title)

    plt.tick_params(direction='in')
    plt.xlabel(labelx, fontsize=14)
    plt.ylabel('Densidad de probabilidad', fontsize=14)
    plt.xlim(0, xlim)  # keV
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=300)
    plt.show()

    
def E_depositada(E, avg=False, max_E=80): # returns the deposited energy 
    e = np.sum(max_E-E)  # 80 keV is the initial energy of the ions
    if avg:
        e /= len(E)
    return e


def Read_QData(filename, skipheader=0, delimiter='\t', comments='#', unpack=True, dtype=float):
    """
    Reads data from a file and returns the data as a tuple of arrays
    """
    # lets read the data as rows with np.loadtxt:
    data = np.genfromtxt(filename, dtype=dtype, comments=comments, skip_header=skipheader, delimiter=delimiter,  unpack=unpack)

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


def Plot_deposited_E_dist(Q, params=None, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, M=400, save=False, title="test"):
    """
    Plots the deposited energy distribution, give a matrix in vector rep Q with the deposited energy at different positions
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
    N = int(np.sqrt(len(Q)))
    Q = np.reshape(Q, (N, N))
    Q = Q.T
    

    # Now lets plot the deposited energy distribution:
    
    v_min = np.min(Q)
    v_max = np.max(Q)
    Q = np.where(Q>0, Q, 0)  # only for visual effects
    # Otra forma mejor era cambiar todos los 0s de Q por v_min y usar un colormap de blanco a rojo

    norm = colors.TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)

    P = ax.imshow(Q, cmap='seismic', origin='lower', aspect='auto', extent=[R0-a, R0+a, -Zx, Zx], norm=norm)    

    cbar = plt.colorbar(P, format='%.2f')
    cbar.set_label(r'$P$ [$MW/m^2$]', fontsize=14)
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

def Plot_separatrix_w_params(params=None, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, M=600, save=False, show=True, title="test"):
    """
    Plots the separatrix of an equilibrium with the associated geometric parameters (pretty picture for the thesis)
    """

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
    plt.contour(R, Z, psi, [0], colors='black')

    fig = plt.gcf()
    fig.set_figwidth(4.2)
    fig.set_figheight(4.2*np.sqrt(2))

    # lets remove the axis
    plt.axis('off')
    # lets plot a horizontal line at z=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    # lets plot a vertical line at r=R0
    plt.axvline(x=R0, ymin=0.05, ymax=0.95, color='black', linestyle='-.', linewidth=1)
    # lets plot a vertical line at r=0
    plt.axvline(x=0, ymin=0.05, ymax=0.95, color='black', linestyle='-', linewidth=1)
    # lets plot an arrow at r=R0
    plt.annotate(s='', xy=(0,-0.025), xytext=(R0-a,-0.025), arrowprops=dict(arrowstyle='<->'))
    # lets add text to the arrow
    plt.text(x=(R0-a)/2-0.02, y=-0.1, s=r'$R_{in}$', fontsize=12)
    # lets plot an arrow at r=Rx
    plt.annotate(s='', xy=(R0-delta*a+0.005,0), xytext=(R0-delta*a+0.005,Zx), arrowprops=dict(arrowstyle='<->'))
    plt.text(x=R0-delta*a+0.0075, y=Zx/2-0.01, s=r'$\kappa a=Z_x$', fontsize=11)
    # lets plot an arrow at r=R0
    plt.annotate(s='', xy=(R0,Zx+0.01), xytext=(R0-delta*a+0.005,Zx+0.01), arrowprops=dict(arrowstyle='<->'))
    plt.text(x=R0-delta*a/2-0.04, y=Zx+0.03, s=r'$\delta a$', fontsize=12)
    # lets plot an arrow at r=R0
    plt.annotate(s='', xy=(R0,-0.025), xytext=(R0+a,-0.025), arrowprops=dict(arrowstyle='<->'))
    plt.text(x=R0+a/2-0.02, y=-0.1, s=r'$a$', fontsize=12)
    # lets plot an arrow at r=Rx
    plt.annotate(s='', xy=(0,0.1), xytext=(R0-delta*a, 0.1), arrowprops=dict(arrowstyle='<->'))
    plt.text(x=(R0-delta*a)/2-0.03, y=0.125, s=r'$R_{x}$', fontsize=12)
    # lets plot an arrow at r=R0
    plt.annotate(s='', xy=(0,-0.3), xytext=(R0, -0.3), arrowprops=dict(arrowstyle='<->'))
    plt.text(x=R0/2-0.03, y=-0.375, s=r'$R_{0}$', fontsize=12)
    # lets plot an arrow at r=R0
    plt.annotate(s='', xy=(0,-0.5), xytext=(R0+a, -0.5), arrowprops=dict(arrowstyle='<->'))
    plt.text(x=R0/2-0.03, y=-0.575, s=r'$R_{out}$', fontsize=12)


    plt.xlim(0, Rmax)
    plt.ylim(Zmin, Zmax)
    plt.axis('scaled')

    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=900)
    elif show:
        plt.show()


def Plot_squareness_fig(params=None, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, M=600, save=False, show=True, title="test"):
    """
    Plots the a diagram to understand the squareness of an equilibrium
    """

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
    plt.contour(R, Z, psi, [0], colors='darkblue')

    fig = plt.gcf()
    fig.set_figwidth(4.2)
    fig.set_figheight(4.2*np.sqrt(2))

    Rx = R0-delta*a
   
    # lets remove the axis
    plt.axis('off') 
    # lets plot the lines:
    plt.plot((Rx, Rx), (0, Zx), 'k-', linewidth=1)
    plt.plot((R0+a, R0+a), (0, Zx), 'k-', linewidth=1)
    plt.plot((Rx, R0+a), (Zx, Zx), 'k-', linewidth=1)
    plt.plot((Rx, R0+a), (0, 0), 'k-', linewidth=1)
    plt.plot((Rx, R0+a), (0, Zx), 'k--', linewidth=1)
    plt.plot((Rx, R0+a), (Zx, 0), 'k-', linewidth=1)
    # lets add the texts/labels
    plt.text(x=Rx-0.05, y=Zx-0.01, s=r'$Z_x$', fontsize=12)
    plt.text(x=Rx-0.05, y=-0.01, s=r'$R_x$', fontsize=12)
    plt.text(x=R0+a+0.01, y=-0.01, s=r'$R_{out}$', fontsize=12)
    Ra = 1.134
    Za = 0.34
    plt.plot((Ra, Rx), (Za, Za), '-', c="gray", linewidth=0.5)
    plt.plot((Ra, Ra), (Za, 0), '-', c="gray", linewidth=0.5)
    plt.text(x=Rx-0.05, y=Za-0.01, s=r'$Z_a$', fontsize=12)
    plt.text(x=Ra-0.015, y=-0.025, s=r'$R_a$', fontsize=12)
    # lets add dots on important intersections
    plt.plot(Ra, Za, 'o', c="red", markersize=4)
    plt.plot((R0+a+Rx)/2, Zx/2, 'o', c="red", markersize=4)
    plt.plot(R0+a, Zx, 'o', c="red", markersize=4)
    # lets name the dots
    plt.text(x=Ra+0.02, y=Za-0, s=r'$B$', fontsize=12)
    plt.text(x=(R0+a+Rx)/2+0.01, y=Zx/2-0.008, s=r'$A$', fontsize=12)
    plt.text(x=R0+a+0.01, y=Zx-0.005, s=r'$C$', fontsize=12)

    # lets add the definition of the squareness
    # plt.text(x=R0, y=0.8*Zx, s=r'$\overline{AB}=s\overline{BC}$', fontsize=12)

    plt.xlim(0.95*Rx, 1.02*(R0+a))
    plt.ylim(-0.02, 1.05*Zx)
    #plt.axis('scaled')

    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=900)
    elif show:
        plt.show()



if __name__ == "__main__":
    # time taken to run the code
    start_time = time.time()
    
    # Plotting
    coefs_dpos = r"PolFlux_DIIID_cand5\s=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.5676_V=3.4043.txt"
    coefs_dneg = r"PolFlux_DIIID_cand5\s=0.0820_alpha_i=-4.5000_alpha_o=2.0000_A=0.5675_V=3.7152.txt"
    # negI = r"C:\Users\fmart\Desktop\Estudio\Balseiro\Tesis\EqTor_puntosX\PolFlux_DIIID_cand5\negative_Ips=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.3547_V=2.3651.txt"

    dr = 0.003222
    dz = 0.004608
    
    # Lets read the Energy deposition file:
    Q_p = Read_QData("Energy_dist_dpos.dat")
    Q_n = Read_QData("Energy_dist_dneg.dat")
    Q_p = np.transpose(Q_p)
    Q_n = np.transpose(Q_n)


    #print(Q.shape)
    #print(Q[3])
    #for q in Q:
    #    Plot_deposited_E_dist(q)
    #Plot_Flux(coefs_dpos, show=False)
    #Plot_deposited_E_dist(np.sum([q for q in Q], axis=0), coefs_dpos)
    #print(len(Q))
    n = int(np.sqrt(len(Q_p[-1])))
    pixel_area_cm = (2*a)/n * (2*Zx)/n * 167**2
    print("área de pixel (cm^2): ", pixel_area_cm)

    BEAM_POWER = 2.5*100*100/pixel_area_cm # MW/m^2

    Q_p_stat = np.sum(Q_p[:-1], axis=0)
    Q_n_stat = np.sum(Q_n[:-1], axis=0)

    Q_p_stat = BEAM_POWER*Q_p_stat/np.sum(Q_p_stat)
    print(np.sum(Q_p_stat))
    Q_n_stat = BEAM_POWER*Q_n_stat/np.sum(Q_n_stat)
    print(np.sum(Q_n_stat))

    Plot_deposited_E_dist(Q_p_stat, coefs_dpos, save=False, title="D_pos_Edepos")
    print(np.min(Q_p_stat))
    print(np.max(Q_p_stat))
    print(np.mean(Q_p_stat))
    
    Plot_deposited_E_dist(Q_n_stat, coefs_dneg, save=False, title="D_neg_Edepos")
    print(np.min(Q_n_stat))
    print(np.max(Q_n_stat))
    print(np.mean(Q_n_stat))
    

    # Plot_separatrix_w_params(coefs_dpos, save=False, show=True, title="Geo_params")
    # Plot_squareness_fig(coefs_dpos, save=False, show=True, title="Squareness")


    # time taken to run the code
    print("--- %s seconds ---" % (time.time() - start_time))
