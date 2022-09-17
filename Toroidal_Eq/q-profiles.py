# FSheffield - 2022
# Finds the q-profile of given analytical MHD equilibria

"""
Negative Triangularity DIII-D equilibrium, cand4 (2), parameters:
M = 0.01
(A=0.567 , V=3.71)
C1 = -0.004892540131, C2 = 0.2072398556, C3 = 0.5556866341, C4 = 0.1702685657, C5 = 0.2663315198
C6 = 0.01729018624, C7 = -0.05564634276, C8 = -0.03619877453
S1 = -0.5395195344810759,  S2 = -0.004129137273799056

Positive Triangularity DIII-D equilibrium, cand4 (2), parameters:
(A = 0.568, V = 3.405)
C1 = 0.001812141542, C2 = 0.4717347435, C3 = 0.5313153095, C4 = 0.4407871704, C5 = 0.4130720551
C6 = 0.02982725318, C7 = -0.04191041218, C8 = 0.0214024026
S1 = -0.6244922921, S2 = -0.02969385881
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Global Eq Parameters
M = 0.01
R0 = 1
k = 1.43
a = 0.67 / 1.67
Zx = k * a
delta = 0.61
Rx = R0 - delta * a

if delta<0: # Negative Triangularity
    S1, S2 = -0.5395195344810759, -0.004129137273799056
    C1, C2, C3, C4, C5, C6, C7, C8 = (-0.004892540131, 0.2072398556, 0.5556866341, 0.1702685657,
                                    0.2663315198, 0.01729018624, -0.05564634276, -0.03619877453)
else:  # Positive Triangularity
    S1, S2 = -0.6244922921, -0.02969385881
    C1, C2, C3, C4, C5, C6, C7, C8 = (0.001812141542, 0.4717347435, 0.5313153095, 0.4407871704, 0.4130720551
                                    , 0.02982725318, -0.04191041218, 0.0214024026)


def Psi(R, Z):
    # Returns the poloidal flux for a given R and Z
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

def B(R, Z):
    # Returns the three components of the magnetic field
    Br = -(C3 * (-8 * R * R * Z) + C4 * (-2 * Z) + C5 * (4 * pow(Z, 3) - 12 * R * R * Z * np.log(R) + 6 * R * R * Z)
        + C6 * (4 * R * R * pow(Z, 3) - 3 * pow(R, 4) * Z) + C7 * (45 * pow(R, 4) * Z * (4 * np.log(R) - 5) / 4 + 30 * R * R * pow(Z, 3) * (1 - 2 * np.log(R)) + 6 * pow(Z, 5))
        + C8 * (6 * R * R * pow(Z, 5) - 15 * pow(R, 4) * pow(Z, 3) + 15 * pow(R, 6) * Z / 4) - S2 * Z) / R

    Bt = np.sqrt(1 + 2 * S2 * Psi(R, Z)) / R

    Bz = (C2 * (2 * R) + C3 * (4 * pow(R, 3) - 8 * R * pow(Z, 2)) + C4 * (R + 2 * R * np.log(R)) + C5 * ((6 * (pow(R, 3) - 2 * R * pow(Z, 2)) * np.log(R) - 6 * pow(R, 3)))
        + C6 * (3 * pow(R, 5) / 4 - 6 * pow(R, 3) * pow(Z, 2) + 2 * R * pow(Z, 4)) + C7 * (15 * (9 * pow(R, 5) - 48 * pow(R, 3) * Z * Z - 2 * (3 * pow(R, 5) - 24 * pow(R, 3) * Z * Z + 8 * R * pow(Z, 4)) * np.log(R)) / 8)
        + C8 * (2 * R * pow(Z, 6) - 15 * pow(R, 3) * pow(Z, 4) + 45 * pow(R, 5) * Z * Z / 4 - 5 * pow(R, 7) / 8) - S1 * R * (1 - np.exp(M * M * R * R)) / (2 * M * M)) / R

    return Br, Bt, Bz

def q(R_, presicion=1e-4):
    # Returns the q safety factor for a given radius
    # "el número de vueltas que hay que efectuar en dirección toroidal 
    # (siguiendo la línea de campo magnético) para efectuar una vuelta en dirección poloidal."

    dtheta = 2*np.pi*presicion  # 2pi/10000
    Br, Bt, Bz = B(R_, 0)  # we start on one side of the poloidal surface and take half a turn
    dr = R_ * dtheta * Br/Bt
    dz = R_ * dtheta * Bz/Bt
    R = R_ + dr
    Z0 = dz 
    theta_counter = 1
    Z = Z0
    while(Z*Z0>0): # while we are still on the same side of the poloidal surface
        Br, Bt, Bz = B(R, Z)  # super slow, it would be best to run this part in c++
        print(Bt, R, Z)
        dr = R * dtheta * Br/Bt
        dz = R * dtheta * Bz/Bt
        R += dr
        Z += dz
        theta_counter += 1
    
    #theta = theta_counter * dtheta
    Toroidal_turns = theta_counter * presicion  # theta / 2pi
    # print(Toroidal_turns)
    Poloidal_turns = 0.5  # half a turn
    return Toroidal_turns/Poloidal_turns



def Plot_Flux(N=400, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, plot_separatrix=False, title="", save=False):
    # Plots the flux surface
    R = np.linspace(Rmin, Rmax, N)
    Z = np.linspace(Zmin, Zmax, N)
    R, Z = np.meshgrid(R, Z)
    psi = np.where(Psi(R, Z)>=0, Psi(R, Z), 0)  # We only plot the postive flux surfaces
    
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

def Plot_safety_factors(rho_1, q_1, rho_2, q_2, title="", save=False):
    # Plots the safety factors of two configurations
    plt.plot(rho_1, q_1, color='red', label=r'$\delta>0$')
    plt.plot(rho_2, q_2, color='blue', label=r'$\delta<0$')

    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$q$')
    plt.title(title)
    plt.tick_params(direction='in')
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
    Plot_Flux(plot_separatrix=False, title="")
    # Lets load the data from Safety_factor_d=pos_2:
    File_pos = "Safety_factor_d=pos_2.txt"
    File_neg = "Safety_factor_d=neg_2.txt"
    R_p, q_p, psi_p = np.loadtxt(File_pos, float, comments='#', delimiter='\t', skiprows=0, unpack=True)
    R_n, q_n, psi_n = np.loadtxt(File_neg, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    rho_p = np.sqrt(1-psi_p/max(psi_p))
    rho_n = np.sqrt(1-psi_n/max(psi_n))
    # Lets now plot rho and q:
    # Plot_safety_factors(rho_p, q_p, rho_n, q_n, title="Safety factors")


    # time taken to run the code
    print("--- %s seconds ---" % (time.time() - start_time))
