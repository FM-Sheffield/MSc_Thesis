// # FSheffield - 2022
// Finds the q-profile of given analytical MHD equilibria
#include <iostream>
#include <iomanip>    
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
constexpr auto PI = 3.14159265358979;

// Global Eq Parameters
const double M = 0.01;
const double R0 = 1;
const double k = 1.43;
const double a = 0.67 / 1.67;
const double Zx = k * a;
const double delta = 0.61;
const double Rx = R0 - delta * a;

// Negative T_2 - DIIID - test cand 4 (A=0.567 , V=3.71):
//const double C1 = -0.004892540131; const double C2 = 0.2072398556; const double C3 = 0.5556866341; const double C4 = 0.1702685657; 
//const double C5 = 0.2663315198; const double C6 = 0.01729018624; const double C7 = -0.05564634276; 
//const double C8 = -0.03619877453; const double S1 = -0.5395195344810759; const double S2 = -0.004129137273799056;

//Positive T_2 - DIIID - test cand 4 (A = 0.568, V = 3.405):
const double C1 = 0.001812141542; const double C2 = 0.4717347435; const double C3 = 0.5313153095; const double C4 = 0.4407871704; 
const double C5 = 0.4130720551; const double C6 = 0.02982725318; const double C7 = -0.04191041218; const double C8 = 0.0214024026;
const double S1 = -0.6244922921; const double S2 = -0.02969385881;



// Returns the poloidal flux for a given R and Z
double Psi_analitico(double R, double Z) {
    double psi2 = R * R;
    double psi3 = pow(R, 4) - 4 * R * R * Z * Z;
    double psi4 = R * R * log(R) - Z * Z;
    double psi5 = pow(Z, 4) - 6 * R * R * Z * Z * log(R) + 3 * R * R * Z * Z + 3 * pow(R, 4) * log(R) / 2.0 - 15 * pow(R, 4) / 8.0;
    double psi6 = R * R * pow(Z, 4) - 3 * pow(R, 4) * Z * Z / 2.0 + pow(R, 6) / 8.0;
    double psi7 = pow(Z, 6) + 15 * R * R * pow(Z, 4) * (1 - 2 * log(R)) / 2.0 + 45 * pow(R, 4) * Z * Z * (2 * log(R) - 2.5) / 4.0 - 15 * pow(R, 6) * (2 * log(R) - 10.0 / 3) / 16.0;
    double psi8 = R * R * pow(Z, 6) - 15 * pow(R, 4) * pow(Z, 4) / 4.0 + 15 * pow(R, 6) * Z * Z / 8.0 - 5 * pow(R, 8) / 64.0;
    double f1 = -(1 + M * M * R * R - exp(M * M * R * R)) * S1 / (4 * pow(M, 4));
    double f2 = -S2 * Z * Z / 2.0;

    return C1 + C2 * psi2 + C3 * psi3 + C4 * psi4 + C5 * psi5 + C6 * psi6 + C7 * psi7 + C8 * psi8 + f1 + f2;
}


// actualiza el valor de B y s_flux con el eq. analitico
void B_Analitico(double R,double Z,double *B){
	double Br = -(C3 * (-8 * R * R * Z) + C4 * (-2 * Z) + C5 * (4 * pow(Z, 3) - 12 * R * R * Z * log(R) + 6 * R * R * Z)
        + C6 * (4 * R * R * pow(Z, 3) - 3 * pow(R, 4) * Z) + C7 * (45 * pow(R, 4) * Z * (4 * log(R) - 5) / 4 + 30 * R * R * pow(Z, 3) * (1 - 2 * log(R)) + 6 * pow(Z, 5))
        + C8 * (6 * R * R * pow(Z, 5) - 15 * pow(R, 4) * pow(Z, 3) + 15 * pow(R, 6) * Z / 4) - S2 * Z) / R;
	double Bt = sqrt(1 + 2 * S2 * Psi_analitico(R, Z)) / R;
	double Bz = (C2 * (2 * R) + C3 * (4 * pow(R, 3) - 8 * R * pow(Z, 2)) + C4 * (R + 2 * R * log(R)) + C5 * ((6 * (pow(R, 3) - 2 * R * pow(Z, 2)) * log(R) - 6 * pow(R, 3)))
        + C6 * (3 * pow(R, 5) / 4 - 6 * pow(R, 3) * pow(Z, 2) + 2 * R * pow(Z, 4)) + C7 * (15 * (9 * pow(R, 5) - 48 * pow(R, 3) * Z * Z - 2 * (3 * pow(R, 5) - 24 * pow(R, 3) * Z * Z + 8 * R * pow(Z, 4)) * log(R)) / 8)
        + C8 * (2 * R * pow(Z, 6) - 15 * pow(R, 3) * pow(Z, 4) + 45 * pow(R, 5) * Z * Z / 4 - 5 * pow(R, 7) / 8) - S1 * R * (1 - exp(M * M * R * R)) / (2 * M * M)) / R;
	B[0]=Br;
	B[1]=Bt;
	B[2]=Bz;
}

double q_profile(double R_, double pres){
    // Returns the q safety factor for a given R

    double dtheta = 2*PI*pres;  // 2pi/10000
    double B[3]; 
    B_Analitico(R_, 0, B); // B[0] = Br, B[1] = Bt, B[2] = Bz
    double dr = R_ * dtheta * B[0]/B[1];
    double dz = R_ * dtheta * B[2]/B[1];
    double R = R_ + dr;
    double Z0 = dz;
    int theta_counter = 1;
    double Z = Z0;
    while(Z*Z0>0){ // while we are still on the same side of the poloidal surface
        B_Analitico(R, Z, B);
        dr = R * dtheta * B[0]/B[1];
        dz = R * dtheta * B[2]/B[1];
        R += dr;
        Z += dz;
        theta_counter += 1;
    }
    //#theta = theta_counter * dtheta
    double Toroidal_turns = 1.0*theta_counter * pres;  //  # theta / 2pi
    //# print(Toroidal_turns)
    double Poloidal_turns = 0.5;  //# half a turn
    return Toroidal_turns/Poloidal_turns;

}



int main(){
    //double R = 1.2;
    //double B[3];
    //cout << "q = " << q_profile(R0-a+0.00001, 0.0001) << endl;

    // Lets create a file to store the data
    // time taken to run the code
    clock_t t;
    t = clock();

    
    ofstream myfile;
    myfile.open ("Safety_factor_d=pos_2.txt");
    myfile << "# R" << "\t" << "q" <<'\t' << "sqrt(flux)"<< endl;
    double R=R0-a+0.0001;  // makes sure the flux is positive
    double Prev_psi = Psi_analitico(R, 0);

    // Lets create an array of logarithmically spaced R values
    double R_array[1000];
    double R_step = (log(R0+a)-log(R))/1000;
    for(int i=0; i<1000; i++){
        R_array[i] = exp(log(R)+i*R_step);
    }
    double psi = Prev_psi;

    // save values until we reach the magnetic axis
    int i = 1;
    while(psi>=Prev_psi){
        myfile << R << "\t" << q_profile(R, 0.00001) <<"\t" << psi << '\n';
        R = R_array[i];
        i+=1;
        Prev_psi = psi;
        psi = Psi_analitico(R, 0);  // elaluate flux at new pos
    }
    myfile.close();

    // time taken to run the code
    t = clock() - t;
    cout << "Runtime: (" << ((float)t)/CLOCKS_PER_SEC << " seconds)." << endl;
    

    return 0;
}
