// EqToroidal.cpp : Basado en https://doi.org/10.1063/5.0027347, incluyendo condiciones para la formación de puntos X y squareness.
// Programa de cálculo numérico para equilibrios de plasma toroidales en tokamaks
// Facundo Sheffield
// 20/10/21

#include <iostream>
#include <iomanip>    
#include <chrono>
#include <fstream>
#include <string>
#include <format>

using namespace std;
constexpr auto PI = 3.14159265358979;

// Initializes normalized parameters
void InitParamsNorm(string Type, double& R_zero, double& a, double& B0, double& Ip, double& k, double& Beta, double& Ri, double& Ro, double& Rt, double& Zt, double delta) {
    if (Type == "ITER") {
        double R_zero_ = 6.2;
        double a_ = 2;
        double B0_ = 5.3;
        double Ip_ = 15;

        R_zero = 1;
        a = 2 / R_zero_;
        B0 = 1;
        Ip = Ip_ * 4 * PI * pow(10, -1) / (B0_ * R_zero_);
        k = 1.7;
        Beta = 0.03;
        Ri = R_zero - a;
        Ro = R_zero + a;
        Rt = R_zero - delta * a;
        Zt = k * a;
    }
    else if (Type == "MAST") {
        double R_zero_ = 0.85;
        double a_ = 0.65;
        double B0_ = 0.84;
        double Ip_ = 2;

        R_zero = 1;
        a = 0.65 / R_zero_;
        B0 = 1;
        Ip = Ip_ * 4 * PI * pow(10, -1) / (B0_ * R_zero_);
        k = 2.4;
        Beta = 0.3;
        Ri = R_zero - a;
        Ro = R_zero + a;
        Rt = R_zero - delta * a;
        Zt = k * a;
    }
    else if (Type == "DIII-D") {
        double R_zero_ = 1.67;  
        double a_ = 0.67;  
        double B0_ = 2.2;
        double Ip_ = 0.9;

        R_zero = 1;
        a = 0.67 / R_zero_;
        B0 = 1;
        Ip = Ip_ * 4 * PI * pow(10, -1) / (B0_ * R_zero_);
        k = 1.43;  // k at X-points
        Beta = 0.015;
        Ri = R_zero - a;
        Ro = R_zero + a;
        Rt = (R_zero - delta * a);  // *1.1 para puntos X si delta es d_95, siguiendo Cerfon-Freidberg (0.9 para R con dpos)
        Zt = k * a;
    }
    else
    {
        cout << "Invalid Type Input\n";
    }
}

// Calculates the solution at (R;Z) for S1 and S2 with X points
double Psi(double R, double Z, double M, double S1, double S2, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za) {
    // Psi_h = c1 + c2*Psi2 + c3*Psi3 + c4*Psi4 + c5*Psi5 + c6*Psi6 + c7*Psi7 + c8*Psi8
    // Psi_p = f1 + f2

    // para agilizar los cálculos se podrían guardar en memoria cosas como log(r) o potencias
    double f1_I = -(1 + M * M * Ri * Ri - exp(M * M * Ri * Ri)) * S1 / (4 * pow(M, 4));
    double f2_I = 0;
    double f1_O = -(1 + M * M * Ro * Ro - exp(M * M * Ro * Ro)) * S1 / (4 * pow(M, 4));
    double f2_O = 0;
    double f1_A = -(1 + M * M * Ra * Ra - exp(M * M * Ra * Ra)) * S1 / (4 * pow(M, 4));
    double f2_A = -S2 * Za * Za / 2;
    double f1_X = -(1 + M * M * Rx * Rx - exp(M * M * Rx * Rx)) * S1 / (4 * pow(M, 4));
    double f1_RZ = -(1 + M * M * R * R - exp(M * M * R * R)) * S1 / (4 * pow(M, 4));
    double f2_X = -S2 * Zx * Zx / 2;
    double f2_RZ = -S2 * Z * Z / 2;

    double psi2_I = Ri * Ri;
    double psi2_O = Ro * Ro;
    double psi2_A = Ra * Ra;
    double psi2_X = Rx * Rx;
    double psi2_RZ = R * R;

    double psi3_I = pow(Ri, 4);
    double psi3_O = pow(Ro, 4);
    double psi3_A = pow(Ra, 4) - 4 * Ra * Ra * Za * Za;
    double psi3_X = pow(Rx, 4) - 4 * Rx * Rx * Zx * Zx;
    double psi3_RZ = pow(R, 4) - 4 * R * R * Z * Z;

    double psi4_I = Ri * Ri * log(Ri);
    double psi4_O = Ro * Ro * log(Ro);
    double psi4_A = Ra * Ra * log(Ra) - Za * Za;
    double psi4_X = Rx * Rx * log(Rx) - Zx * Zx;
    double psi4_RZ = R * R * log(R) - Z * Z;

    double psi5_I = 3 * pow(Ri, 4) * log(Ri) / 2 - 15 * pow(Ri, 4) / 8;
    double psi5_O = 3 * pow(Ro, 4) * log(Ro) / 2 - 15 * pow(Ro, 4) / 8;
    double psi5_A = pow(Za, 4) - 6 * Ra * Ra * Za * Za * log(Ra) + 3 * Ra * Ra * Za * Za + 3 * pow(Ra, 4) * log(Ra) / 2 - 15 * pow(Ra, 4) / 8;
    double psi5_X = pow(Zx, 4) - 6 * Rx * Rx * Zx * Zx * log(Rx) + 3 * Rx * Rx * Zx * Zx + 3 * pow(Rx, 4) * log(Rx) / 2 - 15 * pow(Rx, 4) / 8;
    double psi5_RZ = pow(Z, 4) - 6 * R * R * Z * Z * log(R) + 3 * R * R * Z * Z + 3 * pow(R, 4) * log(R) / 2 - 15 * pow(R, 4) / 8;
    
    double psi6_I = pow(Ri, 6) / 8;
    double psi6_O = pow(Ro, 6) / 8;
    double psi6_A = Ra * Ra * pow(Za, 4) - 3 * pow(Ra, 4) * Za * Za / 2 + pow(Ra, 6) / 8;
    double psi6_X = Rx * Rx * pow(Zx, 4) - 3 * pow(Rx, 4) * Zx * Zx / 2 + pow(Rx, 6) / 8;
    double psi6_RZ = R * R * pow(Z, 4) - 3 * pow(R, 4) * Z * Z / 2 + pow(R, 6) / 8;
    
    double psi7_I = -15 * pow(Ri, 6) * (2 * log(Ri) - 10.0 / 3) / 16;
    double psi7_O = -15 * pow(Ro, 6) * (2 * log(Ro) - 10.0 / 3) / 16;
    double psi7_A = pow(Za, 6) + 15 * Ra * Ra * pow(Za, 4) * (1 - 2 * log(Ra)) / 2 + 45 * pow(Ra, 4) * Za * Za * (2 * log(Ra) - 2.5) / 4 - 15 * pow(Ra, 6) * (2 * log(Ra) - 10.0 / 3) / 16;
    double psi7_X = pow(Zx, 6) + 15 * Rx * Rx * pow(Zx, 4) * (1 - 2 * log(Rx)) / 2 + 45 * pow(Rx, 4) * Zx * Zx * (2 * log(Rx) - 2.5) / 4 - 15 * pow(Rx, 6) * (2 * log(Rx) - 10.0 / 3) / 16;
    double psi7_RZ = pow(Z, 6) + 15 * R * R * pow(Z, 4) * (1 - 2 * log(R)) / 2 + 45 * pow(R, 4) * Z * Z * (2 * log(R) - 2.5) / 4 - 15 * pow(R, 6) * (2 * log(R) - 10.0 / 3) / 16;

    double psi8_I = - 5 * pow(Ri, 8) / 64;
    double psi8_O = - 5 * pow(Ro, 8) / 64;
    double psi8_A = Ra * Ra * pow(Za, 6) - 15 * pow(Ra, 4) * pow(Za, 4) / 4 + 15 * pow(Ra, 6) * Za * Za / 8 - 5 * pow(Ra, 8) / 64;
    double psi8_X = Rx * Rx * pow(Zx, 6) - 15 * pow(Rx, 4) * pow(Zx, 4) / 4 + 15 * pow(Rx, 6) * Zx * Zx / 8 - 5 * pow(Rx, 8) / 64;   
    double psi8_RZ = R * R * pow(Z, 6) - 15 * pow(R, 4) * pow(Z, 4) / 4 + 15 * pow(R, 6) * Z * Z / 8 - 5 * pow(R, 8) / 64;

    // Derivatives
    double dpsi2r_X = 2 * Rx; // Derivative of psi3 with respect to R @ (Rx, Zx)
    double dpsi2r_I = 2 * Ri;
    double dpsi2r_O = 2 * Ro;
    double dpsi3r_X = 4 * pow(Rx, 3) - 8 * Rx * pow(Zx, 2); 
    double dpsi3r_I = 4 * pow(Ri, 3);
    double dpsi3r_O = 4 * pow(Ro, 3);
    double dpsi4r_X = Rx + 2 * Rx * log(Rx);
    double dpsi4r_I = Ri + 2 * Ri * log(Ri);
    double dpsi4r_O = Ro + 2 * Ro * log(Ro);
    double dpsi5r_X = (6 * (pow(Rx, 3) - 2 * Rx * pow(Zx, 2)) * log(Rx) - 6 * pow(Rx, 3));
    double dpsi5r_I = (6 * pow(Ri, 3) * log(Ri) - 6 * pow(Ri, 3));
    double dpsi5r_O = (6 * pow(Ro, 3) * log(Ro) - 6 * pow(Ro, 3));
    double dpsi6r_X = 3 * pow(Rx, 5) / 4 - 6 * pow(Rx, 3) * pow(Zx, 2) + 2 * Rx * pow(Zx, 4);
    double dpsi6r_I = 3 * pow(Ri, 5) / 4;
    double dpsi6r_O = 3 * pow(Ro, 5) / 4;
    double dpsi7r_X = 15 * (9 * pow(Rx, 5) - 48 * pow(Rx, 3) * Zx * Zx - 2 * (3 * pow(Rx, 5) - 24 * pow(Rx, 3) * Zx * Zx + 8 * Rx * pow(Zx, 4)) * log(Rx)) / 8;
    double dpsi7r_I = 15 * (9 * pow(Ri, 5) - 6 * pow(Ri, 5) * log(Ri)) / 8;
    double dpsi7r_O = 15 * (9 * pow(Ro, 5) - 6 * pow(Ro, 5) * log(Ro)) / 8;
    double dpsi8r_X = 2 * Rx * pow(Zx, 6) - 15 * pow(Rx, 3) * pow(Zx, 4) + 45 * pow(Rx, 5) * Zx * Zx / 4 - 5 * pow(Rx, 7) / 8;
    double dpsi8r_I = - 5 * pow(Ri, 7) / 8;
    double dpsi8r_O = - 5 * pow(Ro, 7) / 8;
    double df1r_X = -S1 * Rx * (1 - exp(M * M * Rx * Rx)) / (2 * M * M);
    double df1r_I = -S1 * Ri * (1 - exp(M * M * Ri * Ri)) / (2 * M * M);
    double df1r_O = -S1 * Ro * (1 - exp(M * M * Ro * Ro)) / (2 * M * M);
    double df2r_X = 0;
    double df2r_I = 0;
    double df2r_O = 0;

    double dpsi2z_X = 0;
    double dpsi3z_X = -8 * Rx * Rx * Zx;
    double dpsi4z_X = -2 * Zx;
    double dpsi5z_X = 4 * pow(Zx, 3) - 12 * Rx * Rx * Zx * log(Rx) + 6 * Rx * Rx * Zx;
    double dpsi6z_X = 4 * Rx * Rx * pow(Zx, 3) - 3 * pow(Rx, 4) * Zx;
    double dpsi7z_X = 45 * pow(Rx, 4) * Zx * (4 * log(Rx) - 5) / 4 + 30 * Rx * Rx * pow(Zx, 3) * (1 - 2 * log(Rx)) + 6 * pow(Zx, 5);
    double dpsi8z_X = 6 * Rx * Rx * pow(Zx, 5) - 15 * pow(Rx, 4) * pow(Zx, 3) + 15 * pow(Rx, 6) * Zx / 4;
    double df1z_X = 0;
    double df2z_X = -S2 * Zx;

    double dpsi2zz_I = 0;
    double dpsi2zz_O = 0;
    double dpsi3zz_I = -8 * Ri * Ri;
    double dpsi3zz_O = -8 * Ro * Ro;
    double dpsi4zz_I = -2;
    double dpsi4zz_O = -2;
    double dpsi5zz_I = -12 * Ri * Ri * log(Ri) + 6 * Ri * Ri;
    double dpsi5zz_O = -12 * Ro * Ro * log(Ro) + 6 * Ro * Ro;
    double dpsi6zz_I = -3 * Ri * Ri;
    double dpsi6zz_O = -3 * Ro * Ro;
    double dpsi7zz_I = 45 * pow(Ri, 4) * (4 * log(Ri) - 5) / 4;
    double dpsi7zz_O = 45 * pow(Ro, 4) * (4 * log(Ro) - 5) / 4;
    double dpsi8zz_I = 15 * pow(Ri, 6) / 4;
    double dpsi8zz_O = 15 * pow(Ro, 6) / 4;
    double df1zz_I = 0;
    double df1zz_O = 0;
    double df2zz_I = -S2;
    double df2zz_O = -S2;
    // End of derivatives

    double psi2_io = psi2_I - psi2_O;
    double psi3_io = psi3_I - psi3_O;
    double psi4_io = psi4_I - psi4_O;
    double psi5_io = psi5_I - psi5_O;
    double psi6_io = psi6_I - psi6_O;
    double psi7_io = psi7_I - psi7_O;
    double psi8_io = psi8_I - psi8_O;
    double f1_io = f1_I - f1_O;
    double f2_io = f2_I - f2_O;

    double D3 = (psi2_O * psi3_I - psi2_I * psi3_O) / psi2_io;
    double D4 = (psi2_O * psi4_I - psi2_I * psi4_O) / psi2_io;
    double D5 = (psi2_O * psi5_I - psi2_I * psi5_O) / psi2_io;
    double D6 = (psi2_O * psi6_I - psi2_I * psi6_O) / psi2_io;
    double D7 = (psi2_O * psi7_I - psi2_I * psi7_O) / psi2_io;
    double D8 = (psi2_O * psi8_I - psi2_I * psi8_O) / psi2_io;
    double P1 = (f1_I * psi2_O - f1_O * psi2_I) / psi2_io;
    double P2 = (f2_I * psi2_O - f2_O * psi2_I) / psi2_io;

    double gamm_3A = D3 - psi2_A * psi3_io / psi2_io + psi3_A;
    double gamm_4A = D4 - psi2_A * psi4_io / psi2_io + psi4_A;
    double gamm_5A = D5 - psi2_A * psi5_io / psi2_io + psi5_A;
    double gamm_6A = D6 - psi2_A * psi6_io / psi2_io + psi6_A;
    double gamm_7A = D7 - psi2_A * psi7_io / psi2_io + psi7_A;
    double gamm_8A = D8 - psi2_A * psi8_io / psi2_io + psi8_A;
    double delt_A = P1 + P2 + f1_A + f2_A - psi2_A * (f1_io + f2_io) / psi2_io;

    double gamm_3X = D3 - psi2_X * psi3_io / psi2_io + psi3_X;
    double gamm_4X = D4 - psi2_X * psi4_io / psi2_io + psi4_X;
    double gamm_5X = D5 - psi2_X * psi5_io / psi2_io + psi5_X;
    double gamm_6X = D6 - psi2_X * psi6_io / psi2_io + psi6_X;
    double gamm_7X = D7 - psi2_X * psi7_io / psi2_io + psi7_X;
    double gamm_8X = D8 - psi2_X * psi8_io / psi2_io + psi8_X;
    double delt_X = P1 + P2 + f1_X + f2_X - psi2_X * (f1_io + f2_io) / psi2_io;

    double lamb_3X = dpsi3r_X - dpsi2r_X * psi3_io / psi2_io;
    double lamb_4X = dpsi4r_X - dpsi2r_X * psi4_io / psi2_io;
    double lamb_5X = dpsi5r_X - dpsi2r_X * psi5_io / psi2_io;
    double lamb_6X = dpsi6r_X - dpsi2r_X * psi6_io / psi2_io;
    double lamb_7X = dpsi7r_X - dpsi2r_X * psi7_io / psi2_io;
    double lamb_8X = dpsi8r_X - dpsi2r_X * psi8_io / psi2_io;
    double sigm_X = df1r_X + df2r_X - dpsi2r_X * (f1_io + f2_io) / psi2_io;

    double Det = gamm_3A * gamm_4X * lamb_5X + gamm_4A * gamm_5X * lamb_3X + gamm_5A * gamm_3X * lamb_4X
                - gamm_5A * gamm_4X * lamb_3X - gamm_3A * gamm_5X * lamb_4X - gamm_4A * gamm_3X * lamb_5X;

    double a_36 = (gamm_6A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + gamm_6X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + lamb_6X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;
    double a_37 = (gamm_7A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + gamm_7X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + lamb_7X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;
    double a_38 = (gamm_8A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + gamm_8X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + lamb_8X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;
    double b_3 = (delt_A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + delt_X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + sigm_X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;

    double a_46 = (gamm_6A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + gamm_6X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + lamb_6X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;
    double a_47 = (gamm_7A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + gamm_7X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + lamb_7X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;
    double a_48 = (gamm_8A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + gamm_8X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + lamb_8X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;
    double b_4 = (delt_A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + delt_X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + sigm_X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;

    double a_56 = (gamm_6A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + gamm_6X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + lamb_6X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;
    double a_57 = (gamm_7A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + gamm_7X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + lamb_7X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;
    double a_58 = (gamm_8A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + gamm_8X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + lamb_8X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;
    double b_5 = (delt_A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + delt_X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + sigm_X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;

    double e_26 = -(a_36 * psi3_io + a_46 * psi4_io + a_56 * psi5_io + psi6_io) / psi2_io;
    double e_27 = -(a_37 * psi3_io + a_47 * psi4_io + a_57 * psi5_io + psi7_io) / psi2_io;
    double e_28 = -(a_38 * psi3_io + a_48 * psi4_io + a_58 * psi5_io + psi8_io) / psi2_io;
    double e_20 = -(b_3 * psi3_io + b_4 * psi4_io + b_5 * psi5_io + f1_io + f2_io) / psi2_io;

    double A_2I = dpsi2zz_I - alpha_i * dpsi2r_I;
    double A_3I = dpsi3zz_I - alpha_i * dpsi3r_I;
    double A_4I = dpsi4zz_I - alpha_i * dpsi4r_I;
    double A_5I = dpsi5zz_I - alpha_i * dpsi5r_I;
    double A_6I = dpsi6zz_I - alpha_i * dpsi6r_I;
    double A_7I = dpsi7zz_I - alpha_i * dpsi7r_I;
    double A_8I = dpsi8zz_I - alpha_i * dpsi8r_I;

    double A_2O = dpsi2zz_O - alpha_o * dpsi2r_O;
    double A_3O = dpsi3zz_O - alpha_o * dpsi3r_O;
    double A_4O = dpsi4zz_O - alpha_o * dpsi4r_O;
    double A_5O = dpsi5zz_O - alpha_o * dpsi5r_O;
    double A_6O = dpsi6zz_O - alpha_o * dpsi6r_O;
    double A_7O = dpsi7zz_O - alpha_o * dpsi7r_O;
    double A_8O = dpsi8zz_O - alpha_o * dpsi8r_O;

    double q_11 = e_26 * dpsi2z_X + a_36 * dpsi3z_X + a_46 * dpsi4z_X + a_56 * dpsi5z_X + dpsi6z_X;
    double q_12 = e_27 * dpsi2z_X + a_37 * dpsi3z_X + a_47 * dpsi4z_X + a_57 * dpsi5z_X + dpsi7z_X;
    double q_13 = e_28 * dpsi2z_X + a_38 * dpsi3z_X + a_48 * dpsi4z_X + a_58 * dpsi5z_X + dpsi8z_X;
    double mu_1 = -(e_20 * dpsi2z_X + b_3 * dpsi3z_X + b_4 * dpsi4z_X + b_5 * dpsi5z_X + df1z_X + df2z_X);

    double q_21 = e_26 * A_2O + a_36 * A_3O + a_46 * A_4O + a_56 * A_5O + A_6O;
    double q_22 = e_27 * A_2O + a_37 * A_3O + a_47 * A_4O + a_57 * A_5O + A_7O;
    double q_23 = e_28 * A_2O + a_38 * A_3O + a_48 * A_4O + a_58 * A_5O + A_8O;
    double mu_2 = -(e_20 * A_2O + b_3 * A_3O + b_4 * A_4O + b_5 * A_5O + df1zz_O + df2zz_O - alpha_o * (df1r_O + df2r_O));

    double q_31 = e_26 * A_2I + a_36 * A_3I + a_46 * A_4I + a_56 * A_5I + A_6I;
    double q_32 = e_27 * A_2I + a_37 * A_3I + a_47 * A_4I + a_57 * A_5I + A_7I;
    double q_33 = e_28 * A_2I + a_38 * A_3I + a_48 * A_4I + a_58 * A_5I + A_8I;
    double mu_3 = -(e_20 * A_2I + b_3 * A_3I+ b_4 * A_4I + b_5 * A_5I + df1zz_I + df2zz_I - alpha_i * (df1r_I + df2r_I));

    double Det2 = q_11 * q_22 * q_33 + q_12 * q_23 * q_31 + q_21 * q_13 * q_32 - q_13 * q_22 * q_31 - q_23 * q_32 * q_11 - q_21 * q_12 * q_33;

    double C6 = (mu_1 * (q_22 * q_33 - q_23 * q_32) + mu_2 * (q_13 * q_32 - q_12 * q_33) + mu_3 * (q_12 * q_23 - q_13 * q_22)) / Det2;
    double C8 = (mu_1 * (q_21 * q_32 - q_22 * q_31) + mu_2 * (q_12 * q_31 - q_11 * q_32) + mu_3 * (q_11 * q_22 - q_12 * q_21)) / Det2;
    double C7 = (mu_1 - C8 * q_13 - C6 * q_11) / q_12;
    double C5 = a_56 * C6 + a_57 * C7 + a_58 * C8 + b_5;
    double C4 = a_46 * C6 + a_47 * C7 + a_48 * C8 + b_4;
    double C3 = a_36 * C6 + a_37 * C7 + a_38 * C8 + b_3;
    double C2 = e_26 * C6 + e_27 * C7 + e_28 * C8 + e_20;
    double C1 = C3 * D3 + C4 * D4 + C5 * D5 + C6 * D6 + C7 * D7 + C8 * D8 + P1 + P2;
    
    return C1 + C2 * psi2_RZ + C3 * psi3_RZ + C4 * psi4_RZ + C5 * psi5_RZ + C6 * psi6_RZ + C7 * psi7_RZ + C8 * psi8_RZ + f1_RZ + f2_RZ;
}

// Prints the coefficients of Psi
void PrintPsiCoeffs(double R, double Z, double M, double S1, double S2, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za) {
    // Psi_h = c1 + c2*Psi2 + c3*Psi3 + c4*Psi4 + c5*Psi5 + c6*Psi6 + c7*Psi7 + c8*Psi8
    // Psi_p = f1 + f2

    // para agilizar los cálculos se podrían guardar en memoria cosas como log(r) o potencias
    double f1_I = -(1 + M * M * Ri * Ri - exp(M * M * Ri * Ri)) * S1 / (4 * pow(M, 4));
    double f2_I = 0;
    double f1_O = -(1 + M * M * Ro * Ro - exp(M * M * Ro * Ro)) * S1 / (4 * pow(M, 4));
    double f2_O = 0;
    double f1_A = -(1 + M * M * Ra * Ra - exp(M * M * Ra * Ra)) * S1 / (4 * pow(M, 4));
    double f2_A = -S2 * Za * Za / 2;
    double f1_X = -(1 + M * M * Rx * Rx - exp(M * M * Rx * Rx)) * S1 / (4 * pow(M, 4));
    double f1_RZ = -(1 + M * M * R * R - exp(M * M * R * R)) * S1 / (4 * pow(M, 4));
    double f2_X = -S2 * Zx * Zx / 2;
    double f2_RZ = -S2 * Z * Z / 2;

    double psi2_I = Ri * Ri;
    double psi2_O = Ro * Ro;
    double psi2_A = Ra * Ra;
    double psi2_X = Rx * Rx;
    double psi2_RZ = R * R;

    double psi3_I = pow(Ri, 4);
    double psi3_O = pow(Ro, 4);
    double psi3_A = pow(Ra, 4) - 4 * Ra * Ra * Za * Za;
    double psi3_X = pow(Rx, 4) - 4 * Rx * Rx * Zx * Zx;
    double psi3_RZ = pow(R, 4) - 4 * R * R * Z * Z;

    double psi4_I = Ri * Ri * log(Ri);
    double psi4_O = Ro * Ro * log(Ro);
    double psi4_A = Ra * Ra * log(Ra) - Za * Za;
    double psi4_X = Rx * Rx * log(Rx) - Zx * Zx;
    double psi4_RZ = R * R * log(R) - Z * Z;

    double psi5_I = 3 * pow(Ri, 4) * log(Ri) / 2 - 15 * pow(Ri, 4) / 8;
    double psi5_O = 3 * pow(Ro, 4) * log(Ro) / 2 - 15 * pow(Ro, 4) / 8;
    double psi5_A = pow(Za, 4) - 6 * Ra * Ra * Za * Za * log(Ra) + 3 * Ra * Ra * Za * Za + 3 * pow(Ra, 4) * log(Ra) / 2 - 15 * pow(Ra, 4) / 8;
    double psi5_X = pow(Zx, 4) - 6 * Rx * Rx * Zx * Zx * log(Rx) + 3 * Rx * Rx * Zx * Zx + 3 * pow(Rx, 4) * log(Rx) / 2 - 15 * pow(Rx, 4) / 8;
    double psi5_RZ = pow(Z, 4) - 6 * R * R * Z * Z * log(R) + 3 * R * R * Z * Z + 3 * pow(R, 4) * log(R) / 2 - 15 * pow(R, 4) / 8;

    double psi6_I = pow(Ri, 6) / 8;
    double psi6_O = pow(Ro, 6) / 8;
    double psi6_A = Ra * Ra * pow(Za, 4) - 3 * pow(Ra, 4) * Za * Za / 2 + pow(Ra, 6) / 8;
    double psi6_X = Rx * Rx * pow(Zx, 4) - 3 * pow(Rx, 4) * Zx * Zx / 2 + pow(Rx, 6) / 8;
    double psi6_RZ = R * R * pow(Z, 4) - 3 * pow(R, 4) * Z * Z / 2 + pow(R, 6) / 8;

    double psi7_I = -15 * pow(Ri, 6) * (2 * log(Ri) - 10.0 / 3) / 16;
    double psi7_O = -15 * pow(Ro, 6) * (2 * log(Ro) - 10.0 / 3) / 16;
    double psi7_A = pow(Za, 6) + 15 * Ra * Ra * pow(Za, 4) * (1 - 2 * log(Ra)) / 2 + 45 * pow(Ra, 4) * Za * Za * (2 * log(Ra) - 2.5) / 4 - 15 * pow(Ra, 6) * (2 * log(Ra) - 10.0 / 3) / 16;
    double psi7_X = pow(Zx, 6) + 15 * Rx * Rx * pow(Zx, 4) * (1 - 2 * log(Rx)) / 2 + 45 * pow(Rx, 4) * Zx * Zx * (2 * log(Rx) - 2.5) / 4 - 15 * pow(Rx, 6) * (2 * log(Rx) - 10.0 / 3) / 16;
    double psi7_RZ = pow(Z, 6) + 15 * R * R * pow(Z, 4) * (1 - 2 * log(R)) / 2 + 45 * pow(R, 4) * Z * Z * (2 * log(R) - 2.5) / 4 - 15 * pow(R, 6) * (2 * log(R) - 10.0 / 3) / 16;

    double psi8_I = -5 * pow(Ri, 8) / 64;
    double psi8_O = -5 * pow(Ro, 8) / 64;
    double psi8_A = Ra * Ra * pow(Za, 6) - 15 * pow(Ra, 4) * pow(Za, 4) / 4 + 15 * pow(Ra, 6) * Za * Za / 8 - 5 * pow(Ra, 8) / 64;
    double psi8_X = Rx * Rx * pow(Zx, 6) - 15 * pow(Rx, 4) * pow(Zx, 4) / 4 + 15 * pow(Rx, 6) * Zx * Zx / 8 - 5 * pow(Rx, 8) / 64;
    double psi8_RZ = R * R * pow(Z, 6) - 15 * pow(R, 4) * pow(Z, 4) / 4 + 15 * pow(R, 6) * Z * Z / 8 - 5 * pow(R, 8) / 64;

    // Derivatives
    double dpsi2r_X = 2 * Rx; // Derivative of psi3 with respect to R @ (Rx, Zx)
    double dpsi2r_I = 2 * Ri;
    double dpsi2r_O = 2 * Ro;
    double dpsi3r_X = 4 * pow(Rx, 3) - 8 * Rx * pow(Zx, 2);
    double dpsi3r_I = 4 * pow(Ri, 3);
    double dpsi3r_O = 4 * pow(Ro, 3);
    double dpsi4r_X = Rx + 2 * Rx * log(Rx);
    double dpsi4r_I = Ri + 2 * Ri * log(Ri);
    double dpsi4r_O = Ro + 2 * Ro * log(Ro);
    double dpsi5r_X = (6 * (pow(Rx, 3) - 2 * Rx * pow(Zx, 2)) * log(Rx) - 6 * pow(Rx, 3));
    double dpsi5r_I = (6 * pow(Ri, 3) * log(Ri) - 6 * pow(Ri, 3));
    double dpsi5r_O = (6 * pow(Ro, 3) * log(Ro) - 6 * pow(Ro, 3));
    double dpsi6r_X = 3 * pow(Rx, 5) / 4 - 6 * pow(Rx, 3) * pow(Zx, 2) + 2 * Rx * pow(Zx, 4);
    double dpsi6r_I = 3 * pow(Ri, 5) / 4;
    double dpsi6r_O = 3 * pow(Ro, 5) / 4;
    double dpsi7r_X = 15 * (9 * pow(Rx, 5) - 48 * pow(Rx, 3) * Zx * Zx - 2 * (3 * pow(Rx, 5) - 24 * pow(Rx, 3) * Zx * Zx + 8 * Rx * pow(Zx, 4)) * log(Rx)) / 8;
    double dpsi7r_I = 15 * (9 * pow(Ri, 5) - 6 * pow(Ri, 5) * log(Ri)) / 8;
    double dpsi7r_O = 15 * (9 * pow(Ro, 5) - 6 * pow(Ro, 5) * log(Ro)) / 8;
    double dpsi8r_X = 2 * Rx * pow(Zx, 6) - 15 * pow(Rx, 3) * pow(Zx, 4) + 45 * pow(Rx, 5) * Zx * Zx / 4 - 5 * pow(Rx, 7) / 8;
    double dpsi8r_I = -5 * pow(Ri, 7) / 8;
    double dpsi8r_O = -5 * pow(Ro, 7) / 8;
    double df1r_X = -S1 * Rx * (1 - exp(M * M * Rx * Rx)) / (2 * M * M);
    double df1r_I = -S1 * Ri * (1 - exp(M * M * Ri * Ri)) / (2 * M * M);
    double df1r_O = -S1 * Ro * (1 - exp(M * M * Ro * Ro)) / (2 * M * M);
    double df2r_X = 0;
    double df2r_I = 0;
    double df2r_O = 0;

    double dpsi2z_X = 0;
    double dpsi3z_X = -8 * Rx * Rx * Zx;
    double dpsi4z_X = -2 * Zx;
    double dpsi5z_X = 4 * pow(Zx, 3) - 12 * Rx * Rx * Zx * log(Rx) + 6 * Rx * Rx * Zx;
    double dpsi6z_X = 4 * Rx * Rx * pow(Zx, 3) - 3 * pow(Rx, 4) * Zx;
    double dpsi7z_X = 45 * pow(Rx, 4) * Zx * (4 * log(Rx) - 5) / 4 + 30 * Rx * Rx * pow(Zx, 3) * (1 - 2 * log(Rx)) + 6 * pow(Zx, 5);
    double dpsi8z_X = 6 * Rx * Rx * pow(Zx, 5) - 15 * pow(Rx, 4) * pow(Zx, 3) + 15 * pow(Rx, 6) * Zx / 4;
    double df1z_X = 0;
    double df2z_X = -S2 * Zx;

    double dpsi2zz_I = 0;
    double dpsi2zz_O = 0;
    double dpsi3zz_I = -8 * Ri * Ri;
    double dpsi3zz_O = -8 * Ro * Ro;
    double dpsi4zz_I = -2;
    double dpsi4zz_O = -2;
    double dpsi5zz_I = -12 * Ri * Ri * log(Ri) + 6 * Ri * Ri;
    double dpsi5zz_O = -12 * Ro * Ro * log(Ro) + 6 * Ro * Ro;
    double dpsi6zz_I = -3 * Ri * Ri;
    double dpsi6zz_O = -3 * Ro * Ro;
    double dpsi7zz_I = 45 * pow(Ri, 4) * (4 * log(Ri) - 5) / 4;
    double dpsi7zz_O = 45 * pow(Ro, 4) * (4 * log(Ro) - 5) / 4;
    double dpsi8zz_I = 15 * pow(Ri, 6) / 4;
    double dpsi8zz_O = 15 * pow(Ro, 6) / 4;
    double df1zz_I = 0;
    double df1zz_O = 0;
    double df2zz_I = -S2;
    double df2zz_O = -S2;
    // End of derivatives

    double psi2_io = psi2_I - psi2_O;
    double psi3_io = psi3_I - psi3_O;
    double psi4_io = psi4_I - psi4_O;
    double psi5_io = psi5_I - psi5_O;
    double psi6_io = psi6_I - psi6_O;
    double psi7_io = psi7_I - psi7_O;
    double psi8_io = psi8_I - psi8_O;
    double f1_io = f1_I - f1_O;
    double f2_io = f2_I - f2_O;

    double D3 = (psi2_O * psi3_I - psi2_I * psi3_O) / psi2_io;
    double D4 = (psi2_O * psi4_I - psi2_I * psi4_O) / psi2_io;
    double D5 = (psi2_O * psi5_I - psi2_I * psi5_O) / psi2_io;
    double D6 = (psi2_O * psi6_I - psi2_I * psi6_O) / psi2_io;
    double D7 = (psi2_O * psi7_I - psi2_I * psi7_O) / psi2_io;
    double D8 = (psi2_O * psi8_I - psi2_I * psi8_O) / psi2_io;
    double P1 = (f1_I * psi2_O - f1_O * psi2_I) / psi2_io;
    double P2 = (f2_I * psi2_O - f2_O * psi2_I) / psi2_io;

    double gamm_3A = D3 - psi2_A * psi3_io / psi2_io + psi3_A;
    double gamm_4A = D4 - psi2_A * psi4_io / psi2_io + psi4_A;
    double gamm_5A = D5 - psi2_A * psi5_io / psi2_io + psi5_A;
    double gamm_6A = D6 - psi2_A * psi6_io / psi2_io + psi6_A;
    double gamm_7A = D7 - psi2_A * psi7_io / psi2_io + psi7_A;
    double gamm_8A = D8 - psi2_A * psi8_io / psi2_io + psi8_A;
    double delt_A = P1 + P2 + f1_A + f2_A - psi2_A * (f1_io + f2_io) / psi2_io;

    double gamm_3X = D3 - psi2_X * psi3_io / psi2_io + psi3_X;
    double gamm_4X = D4 - psi2_X * psi4_io / psi2_io + psi4_X;
    double gamm_5X = D5 - psi2_X * psi5_io / psi2_io + psi5_X;
    double gamm_6X = D6 - psi2_X * psi6_io / psi2_io + psi6_X;
    double gamm_7X = D7 - psi2_X * psi7_io / psi2_io + psi7_X;
    double gamm_8X = D8 - psi2_X * psi8_io / psi2_io + psi8_X;
    double delt_X = P1 + P2 + f1_X + f2_X - psi2_X * (f1_io + f2_io) / psi2_io;

    double lamb_3X = dpsi3r_X - dpsi2r_X * psi3_io / psi2_io;
    double lamb_4X = dpsi4r_X - dpsi2r_X * psi4_io / psi2_io;
    double lamb_5X = dpsi5r_X - dpsi2r_X * psi5_io / psi2_io;
    double lamb_6X = dpsi6r_X - dpsi2r_X * psi6_io / psi2_io;
    double lamb_7X = dpsi7r_X - dpsi2r_X * psi7_io / psi2_io;
    double lamb_8X = dpsi8r_X - dpsi2r_X * psi8_io / psi2_io;
    double sigm_X = df1r_X + df2r_X - dpsi2r_X * (f1_io + f2_io) / psi2_io;

    double Det = gamm_3A * gamm_4X * lamb_5X + gamm_4A * gamm_5X * lamb_3X + gamm_5A * gamm_3X * lamb_4X
        - gamm_5A * gamm_4X * lamb_3X - gamm_3A * gamm_5X * lamb_4X - gamm_4A * gamm_3X * lamb_5X;

    double a_36 = (gamm_6A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + gamm_6X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + lamb_6X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;
    double a_37 = (gamm_7A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + gamm_7X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + lamb_7X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;
    double a_38 = (gamm_8A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + gamm_8X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + lamb_8X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;
    double b_3 = (delt_A * (gamm_5X * lamb_4X - gamm_4X * lamb_5X) + delt_X * (gamm_4A * lamb_5X - gamm_5A * lamb_4X) + sigm_X * (gamm_4X * gamm_5A - gamm_4A * gamm_5X)) / Det;

    double a_46 = (gamm_6A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + gamm_6X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + lamb_6X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;
    double a_47 = (gamm_7A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + gamm_7X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + lamb_7X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;
    double a_48 = (gamm_8A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + gamm_8X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + lamb_8X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;
    double b_4 = (delt_A * (lamb_5X * gamm_3X - gamm_5X * lamb_3X) + delt_X * (gamm_5A * lamb_3X - lamb_5X * gamm_3A) + sigm_X * (gamm_5X * gamm_3A - gamm_5A * gamm_3X)) / Det;

    double a_56 = (gamm_6A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + gamm_6X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + lamb_6X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;
    double a_57 = (gamm_7A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + gamm_7X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + lamb_7X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;
    double a_58 = (gamm_8A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + gamm_8X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + lamb_8X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;
    double b_5 = (delt_A * (gamm_4X * lamb_3X - lamb_4X * gamm_3X) + delt_X * (lamb_4X * gamm_3A - gamm_4A * lamb_3X) + sigm_X * (gamm_4A * gamm_3X - gamm_4X * gamm_3A)) / Det;

    double e_26 = -(a_36 * psi3_io + a_46 * psi4_io + a_56 * psi5_io + psi6_io) / psi2_io;
    double e_27 = -(a_37 * psi3_io + a_47 * psi4_io + a_57 * psi5_io + psi7_io) / psi2_io;
    double e_28 = -(a_38 * psi3_io + a_48 * psi4_io + a_58 * psi5_io + psi8_io) / psi2_io;
    double e_20 = -(b_3 * psi3_io + b_4 * psi4_io + b_5 * psi5_io + f1_io + f2_io) / psi2_io;

    double A_2I = dpsi2zz_I - alpha_i * dpsi2r_I;
    double A_3I = dpsi3zz_I - alpha_i * dpsi3r_I;
    double A_4I = dpsi4zz_I - alpha_i * dpsi4r_I;
    double A_5I = dpsi5zz_I - alpha_i * dpsi5r_I;
    double A_6I = dpsi6zz_I - alpha_i * dpsi6r_I;
    double A_7I = dpsi7zz_I - alpha_i * dpsi7r_I;
    double A_8I = dpsi8zz_I - alpha_i * dpsi8r_I;

    double A_2O = dpsi2zz_O - alpha_o * dpsi2r_O;
    double A_3O = dpsi3zz_O - alpha_o * dpsi3r_O;
    double A_4O = dpsi4zz_O - alpha_o * dpsi4r_O;
    double A_5O = dpsi5zz_O - alpha_o * dpsi5r_O;
    double A_6O = dpsi6zz_O - alpha_o * dpsi6r_O;
    double A_7O = dpsi7zz_O - alpha_o * dpsi7r_O;
    double A_8O = dpsi8zz_O - alpha_o * dpsi8r_O;

    double q_11 = e_26 * dpsi2z_X + a_36 * dpsi3z_X + a_46 * dpsi4z_X + a_56 * dpsi5z_X + dpsi6z_X;
    double q_12 = e_27 * dpsi2z_X + a_37 * dpsi3z_X + a_47 * dpsi4z_X + a_57 * dpsi5z_X + dpsi7z_X;
    double q_13 = e_28 * dpsi2z_X + a_38 * dpsi3z_X + a_48 * dpsi4z_X + a_58 * dpsi5z_X + dpsi8z_X;
    double mu_1 = -(e_20 * dpsi2z_X + b_3 * dpsi3z_X + b_4 * dpsi4z_X + b_5 * dpsi5z_X + df1z_X + df2z_X);

    double q_21 = e_26 * A_2O + a_36 * A_3O + a_46 * A_4O + a_56 * A_5O + A_6O;
    double q_22 = e_27 * A_2O + a_37 * A_3O + a_47 * A_4O + a_57 * A_5O + A_7O;
    double q_23 = e_28 * A_2O + a_38 * A_3O + a_48 * A_4O + a_58 * A_5O + A_8O;
    double mu_2 = -(e_20 * A_2O + b_3 * A_3O + b_4 * A_4O + b_5 * A_5O + df1zz_O + df2zz_O - alpha_o * (df1r_O + df2r_O));

    double q_31 = e_26 * A_2I + a_36 * A_3I + a_46 * A_4I + a_56 * A_5I + A_6I;
    double q_32 = e_27 * A_2I + a_37 * A_3I + a_47 * A_4I + a_57 * A_5I + A_7I;
    double q_33 = e_28 * A_2I + a_38 * A_3I + a_48 * A_4I + a_58 * A_5I + A_8I;
    double mu_3 = -(e_20 * A_2I + b_3 * A_3I + b_4 * A_4I + b_5 * A_5I + df1zz_I + df2zz_I - alpha_i * (df1r_I + df2r_I));

    double Det2 = q_11 * q_22 * q_33 + q_12 * q_23 * q_31 + q_21 * q_13 * q_32 - q_13 * q_22 * q_31 - q_23 * q_32 * q_11 - q_21 * q_12 * q_33;

    double C6 = (mu_1 * (q_22 * q_33 - q_23 * q_32) + mu_2 * (q_13 * q_32 - q_12 * q_33) + mu_3 * (q_12 * q_23 - q_13 * q_22)) / Det2;
    double C8 = (mu_1 * (q_21 * q_32 - q_22 * q_31) + mu_2 * (q_12 * q_31 - q_11 * q_32) + mu_3 * (q_11 * q_22 - q_12 * q_21)) / Det2;
    double C7 = (mu_1 - C8 * q_13 - C6 * q_11) / q_12;
    double C5 = a_56 * C6 + a_57 * C7 + a_58 * C8 + b_5;
    double C4 = a_46 * C6 + a_47 * C7 + a_48 * C8 + b_4;
    double C3 = a_36 * C6 + a_37 * C7 + a_38 * C8 + b_3;
    double C2 = e_26 * C6 + e_27 * C7 + e_28 * C8 + e_20;
    double C1 = C3 * D3 + C4 * D4 + C5 * D5 + C6 * D6 + C7 * D7 + C8 * D8 + P1 + P2;

    cout << std::setprecision(10) << "Psi Coeffs:\n" << "C1= " << C1 << '\n' << "C2= " << C2 << '\n' << "C3= " << C3 << '\n' << "C4= " << C4 << '\n' << "C5= " << C5 << '\n' << "C6= " << C6 << '\n'
        << "C7= " << C7 << '\n' << "C8= " << C8 << endl;
    
}

// Calculates |B|^2 based on the poloidal flux at (R,Z)
double B_squared(double R, double Z, double M, double S1, double S2, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za) {
    double h = pow(10, -4);
    double Btheta2 = (2 * S2 * Psi(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za) + 1) / (R * R);  // since B0 is normalized to 1
    double B_R = (Psi(R + h, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za) - Psi(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za)) / h;
    double B_Z = (Psi(R, Z + h, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za) - Psi(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za)) / h;

    return Btheta2 + B_R * B_R + B_Z * B_Z;
}

// Calculates Beta based on eq. (18) (the volume integral)
double Beta_int(double M, double S1, double S2, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za, int N, double dr, double dz) {
    double R = Ri; // R_zero -a
    double Z = -Zx;
    double S = 0;
    double Vol = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double psi = Psi(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za);
            if (psi > 0) {
                S += psi * exp(M * M * R * R) * R / (B_squared(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za));
                Vol += R;
            }
            Z += dz;
        }
        Z = -Zx;
        R += dr;
    }
    S *= 2 * PI * dr * dz;
    Vol *= 2 * PI * dr * dz;

    S = -2 * S1 * S / Vol;
    return S;
}

double printVol(double M, double S1, double S2, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za, int N, double dr, double dz) {
    double R = Ri; // R_zero -a
    double Z = -Zx;
    double Vol = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (Psi(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za) > 0) {
                Vol += R;
            }
            Z += dz;
        }
        Z = -Zx;
        R += dr;
    }
    Vol *= 2 * PI * dr * dz;
    return Vol;
}

double printArea(double M, double S1, double S2, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za, int N, double dr, double dz) {
    double R = Ri; // R_zero -a
    double Z = -Zx;
    double A = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (Psi(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za) > 0) {
                A += 1;
            }
            Z += dz;
        }
        Z = -Zx;
        R += dr;
    }
    A *= dr * dz;
    return A;
}

// Calculates Ip based on eq. (17) (the surface integral)
double IP_int(double M, double S1, double S2, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za, int N, double dr, double dz) {
    double R = Ri;  // R_zero - a
    double Z = -Zx;
    double S = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double psi = Psi(R, Z, M, S1, S2, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za);
            if (psi > 0) {
                S += (-1 * S1 * R * exp(M * M * R * R) + S2 / R);
            }
            Z += dz;
        }
        Z = -Za;
        R += dr;
    }
    S *= dr * dz;
    return S;
}

void NewtonR_2D(double S_[2], int it_num, double M, double alpha_i, double alpha_o, double Ri, double Ro, double Rx, double Zx, double Ra, double Za, int N, double dr, double dz, double Ip, double Beta) {
    double B1 = 0, B2 = 0, a = 0, b = 0, c = 0, d = 0;
    double del = 0, e = pow(10, -4);

    for (int i = 0; i <= it_num; i++) {
        B1 = IP_int(M, S_[0], S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) - Ip;  // f1(guess)
        B2 = Beta_int(M, S_[0], S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) - Beta;  // f2(guees)

        a = (IP_int(M, S_[0] + e, S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) - IP_int(M, S_[0], S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz)) / e;  // d/dS1
        b = (IP_int(M, S_[0], S_[1] + e, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) - IP_int(M, S_[0], S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz)) / e;  // d/dS2
        c = (Beta_int(M, S_[0] + e, S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) - Beta_int(M, S_[0], S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz)) / e;
        d = (Beta_int(M, S_[0], S_[1] + e, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) - Beta_int(M, S_[0], S_[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz)) / e;
        del = b * c - a * d;

        S_[0] -= (b * B2 - d * B1) / del;
        S_[1] -= (c * B1 - a * B2) / del;
        cout << S_[0] << '\t' << endl;
        cout << S_[1] << '\t' << endl;
    }
}




int main()
{
    using clock = std::chrono::system_clock;
    using sec = std::chrono::duration<double>;
    const auto before = clock::now();  // set time
    // set paramenters
    double R_zero = 0, a = 0, B0 = 0, Ip = 0, k = 0, Beta = 0;
    double Ri = 0, Ro = 0, Rx = 0, Zx = 0, Ra = 0, Za = 0;
    double M = 1;
    double delta = -0.61;  // triangularity at X points

    unsigned int s_iterations = 4, alp_o_iterations = 6, alp_i_iterations = 6;
    
    for (unsigned int i_ = 0; i_ < s_iterations; i_++) {
        for (unsigned int j_ = 0; j_ < alp_i_iterations; j_++) {
            for (unsigned int k_ = 0; k_ < alp_o_iterations; k_++) {

            double s = 0.02+2*i_ / 100.0;
            double alpha_i = 0.5 - j_ * 1.0;
            double alpha_o = -0.5 + k_ * 1.0;
            cout << "s= " << s << '\n' << "alpha_i= " << alpha_i << '\n' << "alpha_o= " << alpha_o << '\n';

            // Initialize paramenters
            InitParamsNorm("DIII-D", R_zero, a, B0, Ip, k, Beta, Ri, Ro, Rx, Zx, delta);

            // Cálculo de Ra,Za en función de s:
            double rhoi = sqrt((Rx - Ri) * (Rx - Ri) + Zx * Zx);
            double rhoo = sqrt((Ro - Rx) * (Ro - Rx) + Zx * Zx);

            if(delta < 0){
                Za = Zx * (0.5 + s / rhoi);
                Ra = (Rx + Ri) / 2.0 - s * (Rx - Ri) / rhoi;
            }
            else {
                Za = Zx * (0.5 + s / rhoo);
                Ra = (Rx + Ro) / 2.0 + s * (Ro - Rx) / rhoo;
            }
            cout << "The value for (Ra, Za) with s = " << s << " is: " << Ra << '\t' << Za << endl;
    

            // Initial guess for S1 and S2
            double S[2] = { -0.5, 0.5 };  // {S1, S2}

            int N = 200; // gridsize
            double dr = 2 * a / N;  // = c * a/N, where the domain is  R_zero -c*a/2 < R < R_zero + c*a/2
            double dz = 2 * Zx / N; //-Zx < Z < Zx

            cout << Rx << ' ' << Zx << endl;
            cout << Psi(Ri, 0, M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za) << endl;
            cout << IP_int(M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) << endl;
            cout << Beta_int(M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) << '\n' << endl;

            int itnum = 8;
            NewtonR_2D(S, itnum, M, alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz, Ip, Beta);
            cout << std::setprecision(10) << '\n' << "(S1, S2)= " << S[0] << '\t' << S[1] << endl;

            //PrintPsiCoeffs(Ri, 0, M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za);
            cout << Ip << endl;
            cout << IP_int(M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) << endl;
            cout << Beta_int(M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz) << endl;
            double A = printArea(M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz);
            double V = printVol(M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za, N, dr, dz);
            cout << "Area: " << A << endl;
            cout << "Volume: " << V << endl;

            N = 400;
            dr = 3 * a / N;  // = c * a/N, where the domain is  R_zero -c*a/2 < R < R_zero + c*a/2
            dz = 2.4 * Zx / N; //-1.2Zt < Z < 1.2Zt

            string FileName = "PolFlux_DIIID_cand4_M=1_dneg/", square = "s=", alphai = "alpha_i=", alphao = "alpha_o=", area = "A=", volume = "V=";
            square += format("{:.4f}", s)+'_';
            alphai += format("{:.4f}", alpha_i)+'_';
            alphao += format("{:.4f}", alpha_o) + '_';
            area += format("{:.4f}", A) + '_';
            volume += format("{:.4f}", V);

            FileName += square + alphai + alphao + area + volume + ".txt";
            ofstream oFile(FileName);  // create a file for flux analysis
            //ofstream oFile("PolFlux_DIIID_cand4_M=1_test.txt");
            if (oFile.is_open()) {
                oFile << "# DIIID-like flux equilibrium, M=" << M << ", S1=" << S[0] << ", S2=" << S[1] << ", Rx=" << Rx << ", Zx=" << Zx << ".Below are N, dr, dz, and Zt(Normalized)" << '\n';
                oFile << N << '\t' << dr << '\t' << dz << '\t' << a << '\t' << Zx << '\n';
                for (int m = 0; m < N; m++) {
                    double Z = 1.2 * Zx - m * dz;
                    for (int n = 0; n < N; n++) {
                        double R = R_zero - 1.5 * a + n * dr;
                        oFile << Psi(R, Z, M, S[0], S[1], alpha_i, alpha_o, Ri, Ro, Rx, Zx, Ra, Za) << '\t';
                    }
                }
                oFile.close();
            }
            }
        }
    }

    const sec duration = clock::now() - before;
    std::cout << "Time taken: " << duration.count() << "s" << std::endl;

}


