# Plots the poloidal flux surfaces based on analytical equilibria
import numpy as np
import matplotlib.pyplot as plt
import time
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

# This function plot the magnetic field on the r-z plane:
def plot_B(params, N=400, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, title="", reversed=False):
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    # Create the grid  
    R = np.linspace(Rmin, Rmax, N)
    Z = np.linspace(Zmin, Zmax, N)
    R, Z = np.meshgrid(R, Z)

    # Calculate the magnetic field
    if reversed:
        a, b, c =  B(R, Z, params)
        Br, Bt, Bz = -a, -b, -c
    else:
        Br, Bt, Bz = B(R, Z, params)

    print(Br.max(), Bz.max(), Bt.max())

    # Plot the magnetic field
    plt.streamplot(R, Z, Br, Bz, density=1.5, color='b', linewidth=1)
    plt.tick_params(direction='in')
    plt.xlabel(r'$r/R_0$')
    plt.ylabel(r'$z/R_0$')
    plt.xlim(Rmin, Rmax)
    plt.ylim(Zmin, Zmax)
    plt.axis('scaled')


    plt.title(title)
    plt.tight_layout()
    plt.show()
    

def plot_mod_Bp(params, N=400, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, title="", reversed=False):
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)

    # Create the grid  
    R = np.linspace(Rmin, Rmax, N)
    Z = np.linspace(Zmin, Zmax, N)

    # Calculate the magnetic field
    if reversed:
        a, b, c =  B(R, Z, params)
        Br, Bt, Bz = -a, -b, -c
    else:
        Br, Bt, Bz = B(R, Z, params)

    print(Br.max(), Bz.max(), Bt.max())

    # Plot the magnetic field
    plt.plot(R, np.sqrt(Br**2 + Bz**2), 'b')
    plt.tick_params(direction='in')
    plt.xlabel(r'$r/R_0$')
    plt.ylabel(r'$z/R_0$')
    plt.xlim(Rmin, Rmax)
    plt.ylim(Zmin, Zmax)
    plt.axis('scaled')


    plt.title(title)
    plt.tight_layout()
    plt.show()

def Plot_Flux(params, N=400, Rmin=(R0-1.2*a), Rmax=(R0+1.2*a), Zmin=-1.2*Zx, Zmax=1.2*Zx, plot_separatrix=False, title="", save=False, reversed=False):
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
        plt.contour(R, Z, -psi, 10, cmap='plasma')

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
    else:
        plt.show()

def Plot_eq_Flux(params, N=400, Rmin=(R0-a), Rmax=(R0+a), color="red", label="", title="", norm=True, show=True, save=False):
    """
    Plots the normalized equatorial flux (the flux at z=0)
    """
    # read the params
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)


    R = np.linspace(Rmin, Rmax, N)
    Z = 0
    psi = np.where(Psi(R, Z, params)>=0, Psi(R, Z, params), 0)  # We only plot the postive flux
    if norm:
        psi = psi/np.max(psi)  # normalize the flux
    plt.plot(R, psi, color=color, label=label)

    plt.tick_params(direction='in')
    plt.xlabel(r'$r/R_0$')
    plt.ylabel(r'$\psi/\psi_{max}$')
    plt.xlim(Rmin, Rmax)
    plt.legend(loc='best')
    plt.ylim(0)
    #plt.axis('scaled') looks ugly for this


    #plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png')
    elif show:
        plt.show()


# returns the pressure at a given point for the analytical equilibrium
def pressure(R, Z, params=None):
    # read the params
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)
    
    return -1*params[0]*Psi(R, Z, params)*np.exp(M*M*R*R)


def Plot_T_profile(params, density=0.5, B_0=2, N=400, Rmin=(R0-a), Rmax=(R0+a), color="red", label="", title="", show=True, save=False):
    """
    Plots the temperature profile in keVs. The density profile must be given in units of 10**20 m^-3 and B_0 in Teslas
    """
    # read the params
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)


    R = np.linspace(Rmin, Rmax, N)
    Z = 0
    P = np.where(Psi(R, Z, params)>=0, pressure(R, Z, params), 0)  # We only plot the postive pressure
    
    T_keV = 24.8241698*P*B_0*B_0/density  # in keV
    print("Mean temperature: ", np.mean(T_keV), " keV")

    plt.plot(R, T_keV, color=color, label=label)

    plt.tick_params(direction='in')
    plt.xlabel(r'$r/R_0$')
    plt.ylabel(r'$T~[keV]$')
    plt.xlim(Rmin, Rmax)
    plt.legend(loc='best')
    plt.ylim(0)
    #plt.axis('scaled') looks ugly for this

    #plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png')
    elif show:
        plt.show()

def Plot_pressure_profile(params, N=400, Rmin=(R0-a), Rmax=(R0+a), color="red", label="", title="", show=True, save=False):
    """
    Plots the normalized pressure profile.
    """
    # read the params
    if params is None:
        print("Error: No parameters given")
        return 0
    # if params is a string
    if isinstance(params, str):
        params = np.loadtxt(params, float, comments='#', delimiter='\t', skiprows=0, unpack=True)


    R = np.linspace(Rmin, Rmax, N)
    Z = 0
    P = np.where(Psi(R, Z, params)>=0, pressure(R, Z, params), 0)  # We only plot the postive pressure
    
    print("Mean pressure: ", np.mean(P))

    plt.plot(R, P, color=color, label=label)

    plt.tick_params(direction='in')
    plt.xlabel(r'$r/R_0$')
    plt.ylabel(r'$\hat{P}$')
    plt.xlim(Rmin, Rmax)
    plt.legend(loc='best')
    plt.ylim(0)
    #plt.axis('scaled') looks ugly for this

    #plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png')
    elif show:
        plt.show()

if __name__ == "__main__":
    # time taken to run the code
    start_time = time.time()

    # Plotting
    #Plot_Flux()
    # coefs = (-0.5831152318,0.05676267719,-0.002190157406,0.5580341791,0.6618636006,0.4918686431,0.5099540459,0.02875339697,-0.05279197327,0.02561018266)
    
    # the location of coefficients for the analytical equilibrie:
    coefs_dpos = r"C:\Users\fmart\Desktop\Estudio\Balseiro\Tesis\EqTor_puntosX\PolFlux_DIIID_cand5\s=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.5676_V=3.4043.txt"
    coefs_dneg = r"C:\Users\fmart\Desktop\Estudio\Balseiro\Tesis\EqTor_puntosX\PolFlux_DIIID_cand5\s=0.0820_alpha_i=-4.5000_alpha_o=2.0000_A=0.5675_V=3.7152.txt"
    #negI = r"C:\Users\fmart\Desktop\Estudio\Balseiro\Tesis\EqTor_puntosX\PolFlux_DIIID_cand5\negative_Ips=0.0780_alpha_i=-2.5000_alpha_o=3.5000_A=0.3547_V=2.3651.txt"
    coefs_new_beta_dneg = r"C:\Users\fmart\Desktop\Estudio\Balseiro\Tesis\EqTor_puntosX\PolFlux_DIIID_cand5\Beta=0.0171_s=0.0820_alpha_i=-4.5000_alpha_o=2.0000_A=0.5671_V=3.7117.txt"
    
    # Plot_Flux(negI, plot_separatrix=False, title="", reversed=True)
    #plot_B(coefs_dpos)
    #plot_B(negI)
    #plot_mod_Bp(coefs_dneg)
    
    #Plot_eq_Flux(coefs_dpos, color="red", title="Equatorial flux", label=r"$\delta>0$", norm=False, show=False, save=False)
    #Plot_eq_Flux(coefs_dneg, color="blue", title="Equatorial flux", label=r"$\delta<0$", norm=False, show=False, save=False)
    #plt.savefig("equatorial_Flux" + '.png', dpi=300)
    #plt.show()
    
    #Plot_T_profile(coefs_dpos, color="red", label=r"$\delta>0$", show=False, save=False)
    #Plot_T_profile(coefs_dneg, color="blue", label=r"$\delta<0$", show=False, save=False)
    #Plot_T_profile(coefs_new_beta_dneg, color="green", label=r"$\delta<0$ $\beta=0.0171$", show=False, save=False)

    Plot_pressure_profile(coefs_dpos, color="red", label=r"$\delta>0$", show=False, save=False)
    Plot_pressure_profile(coefs_dneg, color="blue", label=r"$\delta<0$", show=False, save=False)

    plt.savefig("Pressure_prof" + '.png', dpi=200)
    plt.show()
    #Plot_Flux(coefs_dpos, plot_separatrix=False, title="", save=False)
    #Plot_Flux(coefs_new_beta_dneg, plot_separatrix=False, title="", save=False)

    # time taken to run the code
    print("--- %s seconds ---" % (time.time() - start_time))
