# Plots the poloidal flux surfaces based on analytical equilibria
import numpy as np
import matplotlib.pyplot as plt
import time

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


def Plot_Flux(params, N=400, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, plot_separatrix=False, title="", save=False):
    # Plots the poloidal flux surfaces based on the analytical equilibria, the coefficients (params) can be given as a tuple or as a text file
    if params is None:
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    # Plots the flux surface
    R = np.linspace(Rmin, Rmax, N)
    Z = np.linspace(Zmin, Zmax, N)
    R, Z = np.meshgrid(R, Z)
    psi = np.where(Psi(R, Z, params)>=0, Psi(R, Z, params), 0)  # We only plot the postive flux surfaces
    
    plt.contour(R, Z, psi, 10, cmap='plasma')
    # plt.colorbar()
    if plot_separatrix:
        plt.contour(R, Z, psi, [0], colors='black')
    
    fig = plt.gcf()
    fig.set_figwidth(4.2)
    fig.set_figheight(4.2*np.sqrt(2))

    plt.tick_params(direction='in')
    plt.xlabel(r'r/$R_0$')
    plt.ylabel(r'z/$R_0$')
    plt.xlim(Rmin, Rmax)
    plt.ylim(Zmin, Zmax)
    plt.axis('scaled')


    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png')
    else:
        plt.show()


if __name__ == "__main__":
    # time taken to run the code
    start_time = time.time()

    # Plotting
    #Plot_Flux()
    # coefs = (-0.5831152318,0.05676267719,-0.002190157406,0.5580341791,0.6618636006,0.4918686431,0.5099540459,0.02875339697,-0.05279197327,0.02561018266)
    coefs_dpos = "PolFlux_DIIID_cand5_dpos/s=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.5676_V=3.4043.txt"
    coefs_dneg = "PolFlux_DIIID_cand5_dpos/s=0.0820_alpha_i=-4.5000_alpha_o=2.0000_A=0.5675_V=3.7152.txt"
    Plot_Flux(coefs_dpos, plot_separatrix=False, title="")



    # time taken to run the code
    print("--- %s seconds ---" % (time.time() - start_time))
