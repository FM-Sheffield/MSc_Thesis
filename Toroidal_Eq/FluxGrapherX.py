#  Flux grapher
# F Sheffield - 2022

import numpy as np
import matplotlib.pyplot as plt
import os  # there are multiple options
import time

start_time = time.time()  # start timer


def area(vs_):
    """Use Green's theorem to compute the area enclosed by the given contour."""
    a_ = 0
    x0, y0 = vs_[0]
    for [x1, y1] in vs_[1:]:
        dx = x1 - x0
        dy = y1 - y0
        a_ += 0.5 * (y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a_


Directory_Name = "PolFlux_DIIID_cand4_M=1_dneg"
Save_img_directory = "Imágenes\PolFlux_DIIID_cand4_M=1_dneg"

for f in os.listdir(Directory_Name):

    filename = os.fsdecode(f)
    if filename.endswith(".txt"):
        filename = filename.replace(".txt", ".png")

    File = open(Directory_Name + '/' + f, "r")
    # File = open("PolFlux_DIIID_cand4_dneg_2.txt", "r")
    File.readline()  # skip first line
    Y = File.readline()  # Information parameters
    Flux = File.readline()  # Flux data
    File.close()
    Y = np.fromstring(Y, sep='\t')
    Flux = np.fromstring(Flux, sep='\t')
    # np.transpose(Flux)
    # distances are normalized with respect to R_zero (=1)
    N, dr, dz, a, Zt = int(Y[0]), Y[1], Y[2], Y[3], Y[4]
    Flux.resize((N, N))
    r = np.zeros(N)
    z = np.zeros(N)

    for i in range(N):
        r[i] = 1 - 1.5 * a + i * dr
        z[i] = 1.2 * Zt - i * dz
        for j in range(N):
            if Flux[i][j] < 0:
                Flux[i][j] = 0

    R, Z = np.meshgrid(r, z)

    """    i = 0
    for a in Flux[int(N / 2)] / max(Flux[int(N / 2)]):
        if 0.04 < a < 0.06:
            print(r[i])  # value of r where the flux is 5%
        i += 1"""

    fig, ax = plt.subplots(figsize=(4.2, np.sqrt(2) * 4.2))
    CS = ax.contour(r, z, Flux, 10, cmap='plasma')

    # Area inside countour (not working)
    """    countour = CS.collections[0]
    vs = countour.get_paths()[0].vertices
    print(area(vs))
    """
    """
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Plot3D = ax.plot_surface(R, Z, Flux, linewidth=0, color='#871d1d')
    """
    # leg = ax.legend(loc='upper left')
    # plt.annotate(r'$\delta = 0.33$', (1.2, 0.55))
    ax.tick_params(direction='in')
    plt.title('')
    plt.xlabel(r'$R/R_0$')
    plt.ylabel(r'$Z/R_{0}$')

    fig.savefig('../' + Save_img_directory + '/' + filename)
    # plt.show()
    plt.close()

"""
# Flux at R=Rt
Z1 = np.loadtxt("PolFluxRt.txt", float, comments='#', delimiter='\t', skiprows=0, usecols=0)  # Data[0][0] lmbda del 1ero
Psi1 = np.loadtxt("PolFluxRt.txt", float, comments='#', delimiter='\t', skiprows=0, usecols=1)
fig, ax = plt.subplots()  # Energies plot
plt.plot(Z1, Psi1/max(Psi1), color='red', linewidth=1, label="Flujo Poloidal en R=Rt")

leg = ax.legend(loc='upper left')
ax.tick_params(direction='in')
plt.ylim(0)
plt.title('')
plt.xlabel('Z/R_0')
plt.ylabel(r'$\Psi/\Psi_{max}$')
plt.show()
"""

print("--- %s seconds ---" % (time.time() - start_time))
