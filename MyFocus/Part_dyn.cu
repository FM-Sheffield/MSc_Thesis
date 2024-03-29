/*
Dinámica de partículas cargadas en tokamaks con CUDA
Adaptado del código FOCUS -> https://doi.org/10.1016/j.cpc.2018.07.0180010-4655
Facundo Sheffield - 2022
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "Random123/philox.h"  // random numbers in GPU
//#include "Random123/u01.h"  // to get uniform deviates [0,1] (w/cuda7, Random123 ver2017)
#include "Random123/u01fixedpt.h"    // to get uniform deviates [0,1] (w/cuda8)
#include <cuda.h>

#include "curso.h"

#define Z_1			// Deuterium particles
//#define Z_2			// Alpha particles


/* ********** Global parameters. ******************/
// Plasma and reactor parameters  (normalized, DIIID)
// HOST ******************

const double hB_0   = 2.0;				// Campo Magnetico Toroidal en el eje (Tesla)
const double a   = 0.67 / 1.67;			// Radio menor del Toroide (previously a_cm)

// const double pitch_deg = 60; 			// Pitch en grados, para Init_CI_costado
// const int gridsize = 32;				// Gridsize, para Init_CI_costado

const double delta = 0.61;				// equilibrium triangularity
const double k_ = 1.43;					// equilibrium elongation, used to check if a particle has escaped the configuration
const double Zx = k_*a; 				// X-point z coordinate, used to check if a particle has escaped the configuration
const double R   = 1;				// Radio mayor del Toroide (previously R_cm)
const double hR0    = R;                       // Radio normalizado
const double Ep_MeV = .08;                              // Energía del proyectil (inicial, Mev)
const double hmu    = 2.0;                              // fracción masa proyectil/masa proton (creo)
const int    Npart  = 200000;  //            // Numero de partículas
const int    hNstep = 21600000;				// Limite paso temporales. 24000000 @ dt=0.16~>40ms (1ms = 600000 steps)
const int m_steps = 1; 					// number of time steps to measure position
const double hDt    = 0.16;                              // Temporal step (normalized)
const double hZp    = 1.0;                              // Numero atomico proyectil
double hgamma;
const double hta    = 1.0439E-8*hmu/(hZp*hB_0);		// sec. (Campo en Teslas) (ITER: 3.94E-9 s) (PREGUNTAR-> el tiempo de ciclotron)
// constants (Used for adimensionalizations)
const double hc_cgs = 2.9979E10;			// speed of light
const double ha0_cgs = 5.2918E-9;			// Bohr radius
const double hmp_au = 1836.2;				// proton mass (in a.u.)
const double Pi = 3.1415926535897932385;	// Pi

// For Init_Neutral_Beam() initiallization
const double theta_beam = -0.406;  // radians
const double theta_beam_sd = 0.17/6;  // ~9.8°/6
const double z_beam = 0;
// const double z_beam_sd = 0.2*R/1.1;   // elijo tal que 1.05*sigma = 0.2R  (lo ajusto a ojo basado en la data de DIIID de Cesar)
const double z_beam_disp_ang = 10*Pi/180;  // total dispersion angle of the beam in the Z direction
const double tilt_ang = 126*Pi/180.0;  // beam angle in the X-Y plane (rad)

// DEVICE ************************
__device__ double dEp_MeV = Ep_MeV;
__device__ double E_0    = 0.0;				// Campo electrico de referencia
__device__ double B_0    = hB_0;
__device__ double R0     = R;
__device__ double mu_i   = 1.0;				// Fraccion masa ion-target/masa proton. (PREGUNTAR, colisiones?)
__device__ double Zb     = 1.0;				// Atomic number of Plasma ions 
__device__ double n_e    = 1.0;				// Densidad e- y p+ del Plasma (CORE) (10E14 cm-3)
__device__ double nH     = 0.01;			// Hydrogen impurities (10E14 cm-3)
__device__ double nH2    = 0.005;			// Molecular H impurity (10E14 cm-3)
__device__ double nHe    = 0.005;			// Idem Helio.
__device__ int    Nstep  = hNstep;			// Limite pasos temporales
//__device__ int    Nec	 = 100;			// Cada cuanto computo las colisiones elásticas.
__device__ double Zp     = hZp;				// Numero atomico proyectil
__device__ double mu     = hmu;             // fraccion masa proyectil/masa proton
__device__ double Dt     = hDt;				//
__device__ double da_cm  = a;
__device__ double ta     = hta;
__device__ __constant__ double PI = 3.1415926535897932385;
__device__ double c_cgs = hc_cgs;
__device__ double a0_cgs = ha0_cgs;
__device__ double mp_au = hmp_au;

 struct Part {
	double E_keV; 			// Energia en keV. (los files estan en esta unidad).
	int Z;  				// Numero atomico.
	int q;					// carga neta. (Adimensional)		
	double r[3];			// posición. Coord. cilindricas (r, theta, z). (Adimensional)
	double v[3];			// velocidad. Coord. cilindricas. (Adimensional)
	double time;			// time of particle evolution.
	int state;				// -2 = neutra; -1 = sin determinar; 0 = escapada; 1 = banana; 2 = clockwise; 3 = anticlockwise; 4 = outlier
	int sense;				// sense of rotation
    double pitch;  			// Vparalela al campo (V_par/V=cos(pitch))
    double flux;
    int flag; 				// Indica algún flag, en este caso es 1 si salió y volvió a entrar y 0 else

	//double Ionization_data[5];  // saves certain data at ionization. In order, [r, theta, z, v_pll/v, E_kev] 
	double Escaped_data[5];    // saves data (r, th, z, E_kev, time) of a particle when it escapes

	double Diagnosis_data[10][10];  // saves certain data of particles for further d	iagnosis at different times. 
							     // In order, Time_it*[r, theta, z, vr, vth, vz, pitch, E_kev, state, time]

	#ifdef Z_1			
	int n;				// for Deuterium: quantum number, for inelastic col
	double timeAt;      //(IN SEC.)	// for Deuterium: time in AU
	#endif
	};

#include "Magnetic_field.h"
#include "General_functions.h" // Utiliza "struct Part" y los #define a_cm, mu, etc!! -> Por eso el include está luego de struct Part
#include "Elastic_collision_module.h"  // Módulo de colisiones elásticas
#include "Inelastic_collision_module.h"  // Módulo de colisiones ineslásticas

#define NQ 70  // size of de Matrix Q (d_Q) used for the energy distribution

// Control Trayectoria:
struct Position {
	double r[3];
	double rg[3];
	};
// ----------------

//Control Trayectoria/Evolución temporal: ---------------------------
__device__ void global_2D_add(double *Q, int Nr, int Nz, struct Part * d_He, double aux); 
	// ^^Suma una cantidad aux a la matriz Q[Nr][Nz] en la posición (r,z) de la particula d_He, para calculo de distribuciones 2D
__global__ void Evolution (struct Part * d_He, int Npart, long init, double * d_Q, double *d_Q_container) {
	//Evolución temporal "normal", asigna los tipos de órbitas en d_He.state

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	int diag_time[9] = {600000, 2400000, 2*2400000, 3*2400000, 4*2400000, 5*2400000, 6*2400000, 7*2400000, 8*2400000}; // times (iterations) at which to save data for diagnosis (except the last one)
	
	int Nec=80000;		// steps for elastic collisions - 80k is okay for E=80keV
	int next_col=Nec;	// total steps for next collision

	int Nic = 2; 	//steps for inelastic collisions
	int n = 0;
	int i,kk;	//semilla random numbers
	int q1 = 2;
	double qq00,qq11,omega=16.e-4,tiempo,tiempo0; //flag
	double s_flux, EkeV0;

	double y;// gamma

	//sense variables -----------
	double t0 = 0.0;
	double vpar, vpar_t;
	double B[3], E[3];
	// y = sqrt(10*Ep_MeV) * 0.01758437;  // gamma DIIID, 0.0175 es para 100 keV @ 2.2T
	y = sqrt(10*Ep_MeV) * 0.019342807;  // gamma DIIID, 0.0193 es para 100 keV @ 2T
	double initial_proyection=0.0,proyection=0.0;
	//Control Trayectoria: ------- 
	int j = 0;

	kk=0;
	short unsigned int Period_tol = 25;  // N° de pasos temporales que puede estar fuera del eq (~1 períodos)
	// Period_tol = 0 si los perfiles de n y T no están definidos afuera y quiere usar colisiones, sino se podría pedir que no hayan col
	// fuera de la separatriz, pero no tiene mucho sentido agregarlo porque hay efectos de SOL no tenidos en cuenta.

	// Colisiones elásticas:

	// Random numbers C :
	// Random numbers initialization -------------
	philox2x32_ctr_t   c={{}}; 
	philox2x32_ukey_t  uk = {{}}; 
	uk.v[0] = id + (int)init;		 
	philox2x32_key_t   k = philox2x32keyinit(uk); 
	philox2x32_ctr_t   p;	 
	double Ran1,Ran2; 
	double Ran_EC[4];    // pitch-energy 
	//double Ran_EC[6];  // euler	

	for(i=0;i<1000;i++){ 
	 	c.v[0] = i; 
	 	p = philox2x32(c, k); 
	} 


	if(id < Npart) {
		n = 0; 		
		int diag_it = 0; // current diagnosis iteration

		// ya están en las condiciones iniciales	
		//initial_proyection = Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
		//d_He[id].flux=s_flux;

		initial_proyection = d_He[id].pitch;
		//printf("proyection= %e\n", initial_proyection);  //proyection= 9.000147e-01 para debuggear

		short unsigned int out_counter = 0;  // if out_counter = Period_tol -> escapada
		bool is_in = true;
		bool was_outside = false;


		do{ 
			
			// Note: diag_it could land out of bounds. Be careful with that.
			if (n==diag_time[diag_it]) {
				//printf("diag_it = %d\n", diag_it);
				d_He[id].Diagnosis_data[diag_it][0] = d_He[id].r[0];
				d_He[id].Diagnosis_data[diag_it][1] = d_He[id].r[1];
				d_He[id].Diagnosis_data[diag_it][2] = d_He[id].r[2];
				d_He[id].Diagnosis_data[diag_it][3] = d_He[id].v[0];
				d_He[id].Diagnosis_data[diag_it][4] = d_He[id].v[1];
				d_He[id].Diagnosis_data[diag_it][5] = d_He[id].v[2];
				d_He[id].Diagnosis_data[diag_it][6] = d_He[id].pitch;  // NOTE: this is not the actually the pitch! its v_par
				d_He[id].Diagnosis_data[diag_it][7] = d_He[id].E_keV;
				d_He[id].Diagnosis_data[diag_it][8] = d_He[id].state;
				d_He[id].Diagnosis_data[diag_it][9] = d_He[id].time;
				
				// copy the (diag_it)th Q matrix
				for (int i = 0; i < NQ; i++) {
						for (int j = 0; j < NQ; j++) {
							d_Q_container[(NQ*NQ*diag_it)+(i*NQ + j)] = d_Q[i*NQ + j];  // careful with this, d_Q is being updated in parallel
						}
					}
				diag_it++;
			}


			RK46_NL(d_He+id, y);
			//Boris_c(d_He+id, y);
			n++;

			// Inelastic collisions: ----------------------------------------
			/*if(n%Nic == 0){  // según hugo
				i = i + 1;
				c.v[0] = i;
				i = i + 1;
				c.v[1] = i;
				p = philox2x32(c, k);
				Ran1 = (double) u01_closed_closed_32_53(p[0]);
				Ran2 = (double) u01_closed_closed_32_53(p[1]);

				Inelastic_collisions(d_He+id,(double)Nic*Dt*ta, Ran1);

				//if(q1 != d_He[id].q){
				//	Nprocess = Nprocess + 1;
				//	q1 = d_He[id].q;
				//}
			}  */

			
			if(n%Nic == 0 && d_He[id].q == 0 && s_flux > 0){  
				Inelastic_collisions(d_He+id,(double)Dt*Nic*ta, &i, (int)init, id);
				if (d_He[id].q != 0){ 
					d_He[id].state = -1; // ionized
				}
				/*
				// saves data at ionization
				// NOTA: En realidad así cómo está el pitch está un paso atrás, asumo que no afecta significativamente
				if (d_He[id].q != 0){  
					d_He[id].Ionization_data[0] = d_He[id].r[0];
					d_He[id].Ionization_data[1] = d_He[id].r[1];
					d_He[id].Ionization_data[2] = d_He[id].r[2];
					d_He[id].Ionization_data[3] = d_He[id].pitch/(d_He[id].v[0]*d_He[id].v[0]+d_He[id].v[1]*d_He[id].v[1]+d_He[id].v[2]*d_He[id].v[2]);  // v_pll/v
					d_He[id].Ionization_data[4] = d_He[id].E_keV;
				}
				*/
				//printf("carga: %d\n", d_He[id].q);

				//if(q1 != d_He[id].q){
				//	Nprocess = Nprocess + 1;
				//	q1 = d_He[id].q;
				//}
			}

  			// Elastic collisions: ------------------------------------------
			// faltaría chequear estos valores de Nec con el cálculo analítico del tiempo de frenamiento para Deuterio
			

			if(d_He[id].E_keV > 1000){
			      	Nec = 5000;
			      }else if(d_He[id].E_keV <= 1000.0 && d_He[id].E_keV > 100.0){
			      	Nec = 1000;
			      }else if(d_He[id].E_keV <= 100.0 && d_He[id].E_keV > 10.0){
			      	Nec = 500;
			      }else {
			      	Nec = 100;
			      }
			      if(n%Nec == 0){
					EkeV0 = d_He[id].E_keV;
					Elastic_collisions(d_He+id, 1.0*Nec*Dt, &i, init, id);
					global_2D_add(d_Q, NQ, NQ, d_He+id, (EkeV0 - d_He[id].E_keV));
				  }
			/*
			if(n == next_col){
				Elastic_collisions(d_He+id, 1.0*Nec*Dt, &i, init, id);

				// Recalculo Nec luego de c/ colisión
				Nec = 80000*pow(d_He[id].E_keV/80.0, 1.5);  // Va como T^3/2, para 1 kev debería ser 700 veces más chico que para 80
				next_col += Nec;
				// Randoms numbers needed in elast. collision -------
				/*
				i = i + 1;
				c.v[0] = i;		// Some loop-dependent application
				i = i + 1;
				c.v[1] = i;		// another loop-dependent application variable.
				p = philox2x32(c, k);
				Ran_EC[0] = (double) u01_open_open_32_53(p[0]);
				Ran_EC[1] = (double) u01_open_open_32_53(p[1]);
				//Ran_gauss(&Ran_EC[0]); // Activar para euler
				i = i + 1;
				c.v[0] = i;		
				i = i + 1;
				c.v[1] = i;	
				p = philox2x32(c, k);
				Ran_EC[2] = (double) u01_open_open_32_53(p[0]);
				Ran_EC[3] = (double) u01_open_open_32_53(p[1]);
				Ran_gauss(&Ran_EC[2]);
				i = i + 1;
				c.v[0] = i;		
				i = i + 1;
				c.v[1] = i;	
				p = philox2x32(c, k);
				Ran_EC[4] = (double) u01_open_open_32_53(p[0]);
				Ran_EC[5] = (double) u01_open_open_32_53(p[1]);
				Ran_gauss(&Ran_EC[4]);
				* /

				// __device__ void Elastic_collisions_SV_euler (struct Part *He, double Dt, double * Ran_EC);
				//Elastic_collisions_PE_MC(d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				//Elastic_collisions_euler (d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				// (Notar que para euler necesito 6 nros gaussianos!!!!)
			}
			*/
			// ----------------------------------------------------------------
			// nueva vparalela
			proyection=Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
			/*if(n==3){
				printf("proyection= %e\n", proyection);  // for debug
			}*/
			d_He[id].flux=s_flux;

			if(d_He[id].r[0]<0.8*(R-a) || d_He[id].r[0]>1.2*(R+a) || d_He[id].r[2]<-1.2*Zx || d_He[id].r[2]>1.2*Zx){
				d_He[id].state=0;  // escapada, fuera de la configuración
				// We save the coordinated of the escaped particle
				d_He[id].Escaped_data[0] = d_He[id].r[0];
				d_He[id].Escaped_data[1] = d_He[id].r[1];
				d_He[id].Escaped_data[2] = d_He[id].r[2];
				d_He[id].Escaped_data[3] = d_He[id].E_keV;
				d_He[id].Escaped_data[4] = d_He[id].time;
			}

			if(d_He[id].q != 0){  // no aplica a particulas neutras
				if(s_flux<0){  
					was_outside = true;
					is_in = false;
					
					if (out_counter == Period_tol) {
						d_He[id].state=0;  // escapada
						// We save the coordinated of the escaped particle
						d_He[id].Escaped_data[0] = d_He[id].r[0];
						d_He[id].Escaped_data[1] = d_He[id].r[1];
						d_He[id].Escaped_data[2] = d_He[id].r[2];
						d_He[id].Escaped_data[3] = d_He[id].E_keV;
						d_He[id].Escaped_data[4] = d_He[id].time;
						break;
					}
					out_counter++;
					
				} else {
					is_in = true;
					out_counter=0;
					if(was_outside && is_in){
						d_He[id].flag = 1;
					}
				}

				if(n>100000 && (proyection*initial_proyection)<0){  // puedo ponerle más condiciones para determinar mejor las órbitas
					d_He[id].state=1;  // banana
					// break;  // puedo comentar el break para ver la órbita completa
				}
			}
			d_He[id].pitch=proyection;
			
		}while(n<Nstep && d_He[id].state != 0 );

		if (d_He[id].E_keV > 0){
			if (d_He[id].state == -1){  // sin asignar 
				if(d_He[id].pitch>0){
					d_He[id].state = 2;  // Clockwise
				} else if(d_He[id].pitch<0){
					d_He[id].state = 3;  // Anticlockwise
				} else {
					d_He[id].state = 4;  // Outlier!
				}
			} 
		} else {
			printf("ERROR: E_keV = %f\n", d_He[id].E_keV);
			d_He[id].state = 4;  // Outlier, energía cinética negativa o NaN
		}

		// Save final data
		d_He[id].Diagnosis_data[9][0] = d_He[id].r[0];
		d_He[id].Diagnosis_data[9][1] = d_He[id].r[1];
		d_He[id].Diagnosis_data[9][2] = d_He[id].r[2];
		d_He[id].Diagnosis_data[9][3] = d_He[id].v[0];
		d_He[id].Diagnosis_data[9][4] = d_He[id].v[1];
		d_He[id].Diagnosis_data[9][5] = d_He[id].v[2];
		d_He[id].Diagnosis_data[9][6] = d_He[id].pitch;
		d_He[id].Diagnosis_data[9][7] = d_He[id].E_keV;
		d_He[id].Diagnosis_data[9][8] = d_He[id].state;
		d_He[id].Diagnosis_data[9][9] = d_He[id].time;

		// copy the 10th Q matrix
		for (int i = 0; i < NQ; i++) {
				for (int j = 0; j < NQ; j++) {
					d_Q_container[(9*NQ*NQ)+(i*NQ + j)] = d_Q[i*NQ + j];  // careful with this, d_Q is being updated in parallel
				}
			}
	}
}


__device__ void global_2D_add(double *Q, int Nr, int Nz, struct Part * d_He, double aux){
	// Suma una cantidad aux a la matriz Q[Nr][Nz] en la posición (r,z) de la particula d_He, para calculo de distribuciones 2D
	// Esta función va a evaluar distribuciones 2D (poloidal)
	// en las que mas de un thread puede acceder a un elemento
	// de la matriz ``al mismo tiempo''.
	// Usamos operaciones atómicas para evitar el overlap.

	int i,j;

	double dr = (2*a)/((double)Nr-1.0);  
	double dz = (2*Zx)/((double)Nz-1.0);

	i = floor(((*d_He).r[0] - (R0-a))/dr);
	j = floor(((*d_He).r[2] + Zx)/dz);

	//es natural que hayan muchos i, j iguales.
	//Entonces, la acumulación 'tipo CPU': 
	//Q[i*Nz+j] = Q[i*Nz+j]+1.0;
	//es incorrecta.

	//Usando sm_60 en adelante con cuda8, es posible usar 
	//operaciones atómicas con double.
	atomicAdd(&(Q[i*Nz+j]),aux);

}


int main(){
	// Nota: en realidad xx = r, yy=theta, zz=z
    double Ipos[3], Ivel[3], tiempo, s_flux;
	double xx[Npart],yy[Npart],zz[Npart],vx[Npart],vy[Npart],vz[Npart];
	
	struct Part He[Npart];
	double Q[NQ][NQ] = {0};			// matrix for energy deposition
	double Q_container[10*NQ*NQ]={0};	// container for Q matrix at different times

	int ip;         			// Particle index
	double x,r;				// Initial x position (initializated in fn init_r)
	double rg[3];				// Guiding center.

	int Part_charge[3]={0,0,0};		// Final charge state counter.
	
	struct timeval start;			// Computational time.
	struct timeval finish; 
	double elapsed_time;
	
	/* *********** Output files  ***************/
	FILE *File_IC = fopen("time0000.dat","w");  // Initial Conditions
	if(File_IC == NULL){
		printf("Error File_IC");
		exit(1);}

	/*FILE *File_FC = fopen("time0001.dat","w");  // Final Conditions
	if(File_FC == NULL){
		printf("Error File_FC");
		exit(1);}
	*/

	FILE *File_St = fopen("SR_1MeV0_0x20_He_e_euler_dpos.dat","w");  // stats
	if(File_St == NULL){
		printf("Error File_St");
		exit(1);}

	FILE *File_Orbit_types = fopen("Orbits_dpos.dat","w");  // Conteo de órbitas
	if(File_Orbit_types == NULL){
		printf("Error File_Orbit_types");
		exit(1);}

	/*
	FILE *File_Ion = fopen("ionization.dat","w");  // data at ionization
	if(File_Ion == NULL){
		printf("Error File_Ion");
		exit(1);}
	fprintf(File_Ion,"# Particle stats at ionization.\n# R; theta; z; v_pll/v; E_kev\n");
	*/
	FILE *File_Esc = fopen("escapadas_dpos.dat","w");  // escaped particle coordinates
	if(File_Esc == NULL){
		printf("Error File_Esc");
		exit(1);}
	fprintf(File_Esc,"# Escaped particle coordinates and Energy (KeV).\n# R; theta; z; Energy; time\n");

	FILE *File_Diag = fopen("Diagnostics_dpos.dat","w");  // escaped particle coordinates
	if(File_Diag == NULL){
		printf("Error File_Esc");
		exit(1);}
	fprintf(File_Diag,"# Several particle properties at 10 different times.\n# Time_it; R; theta; z; vr; vth; vz; pitch; E_kev; state; time\n");

	FILE *File_Edist = fopen("Energy_dist_dpos.dat","w");  // energy deposition at different times
	if(File_Edist == NULL){
		printf("Error File_Edist");
		exit(1);}
	fprintf(File_Edist,"# Energy distribution matrix from R_i to R_o and -Zx to Zx at different times\n#dr=%f\tdz=%f\n",
			(2*a)/((double)NQ-1.0),(2*Zx)/((double)NQ-1.0));

	/*********************************************/ 

	/* Random numbers initialization *****/
	double f;
	time_t tran = time(NULL);
	init = labs(init - tran);
	//init = labs(init - 10);  //same velocity for each particle on every iteration
	long *ptrinit = &init;
	for(ip=0;ip<1000;ip++)
		f = ran2(ptrinit);		

	/* ***** Particle Initialization *********/

	//Init_rv(&xx[0],&yy[0],&zz[0],&vx[0],&vy[0],&vz[0],&tiempo,Npart);
	//Init_CI_costado(&xx[0],&yy[0],&zz[0],&vx[0],&vy[0],&vz[0], pitch_deg, gridsize, delta);

	Init_Neutral_Beam(He, theta_beam, theta_beam_sd, z_beam, z_beam_disp_ang, Ep_MeV, tilt_ang);

	// Save initial conditions
	fprintf(File_IC,"Número - tiempo - r - theta - z - Vr - Vtheta - Vz - E (kev) - psi - vparalela - sentido\n");
	for(ip=0;ip<Npart;ip++){
		fprintf(File_IC,"%d %f %f \t %f \t %f \t %f \t %f \t %f %f %f %f %d\n",ip,He[ip].time, He[ip].r[0], He[ip].r[1],He[ip].r[2],
				He[ip].v[0], He[ip].v[1], He[ip].v[2],He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);
	}


	//double hgamma = sqrt(10 * Ep_MeV) * 0.01758437;
	double hgamma = sqrt(10*Ep_MeV) * 0.019342807;  // B=2T

	printf("E in: %.14f keV \n", He[0].E_keV);
	// printf("x in: %.14f \n", x);
	
	/* ***** Output Statistical ****** */
	fprintf(File_St, "gamma: \t %f \n", sqrt(10*Ep_MeV) * 0.019342807);
	fprintf(File_St,"Initial \n");
	fprintf(File_St, "Dt: \t %f \n", hDt);
	fprintf(File_St, "Nstep: \t %d \n", hNstep);
	fprintf(File_St, "Simul time: \t %f msec. \n", (double)hNstep*hDt*hta*1000.0);
	fprintf(File_St, "Npart: \t %d \n", Npart);
	fprintf(File_St, "Z beam: \t %d \n", (int)hZp);
	fprintf(File_St, "Ep: \t %f keV \n", Ep_MeV*1000.0);
	fprintf(File_St, "x: \t %f \n", x);


	/* ******** Particle evolution **********/
	
	gettimeofday(&start,NULL);

	gettimeofday(&finish,NULL);
	

	elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)/1.0e6;
	printf("Elapsed time before evolution: \t %f sec.\n", elapsed_time);

	/***** CUDA ******/
	load_atomic_processes();	// loads the data from the collision matrices to the GPU

	struct Part *d_He;
	double *d_Q;			// matrix for energy deposition
	double *d_Q_container;

	HANDLE_ERROR(cudaMalloc( (void**) &d_He, Npart*sizeof(Part) ));
    HANDLE_ERROR(cudaMemcpy( d_He, &He, Npart*sizeof(Part), cudaMemcpyHostToDevice));
	checkCUDAError("Particle copy: failed \n");
	HANDLE_ERROR(cudaMalloc( (void**) &d_Q, NQ*NQ*sizeof(double) ));
	HANDLE_ERROR(cudaMemcpy( d_Q, &Q, NQ*NQ*sizeof(double), cudaMemcpyHostToDevice )); 
	checkCUDAError("Q copy: failed \n");
	HANDLE_ERROR(cudaMalloc( (void**) &d_Q_container, 10*NQ*NQ*sizeof(double) ));
	HANDLE_ERROR(cudaMemcpy( d_Q_container, &Q_container, 10*NQ*NQ*sizeof(double), cudaMemcpyHostToDevice )); 
	checkCUDAError("Q copy: failed \n");
	
    int dev = 0;
        if(cudaGetDevice(&dev)!= cudaSuccess)
                printf("cudaGetDeviceCount FAILED");

        cudaDeviceProp deviceProp;
        //for(int dev = 0; dev < deviceCount; ++dev){
                cudaGetDeviceProperties(&deviceProp,dev);
                printf("\nPlaca %d: %s \n", dev, deviceProp.name);
	int numthreads = 32;  // 32, se puede probar con otras potencias de 2 para optimizar
	// two threads from different blocks cannot cooperate
	int numblocks = (Npart+numthreads-1)/numthreads;  // recordar que hay límite máximo de bloques
	dim3 block_size(numthreads);
  	dim3 grid_size(numblocks);

	//Control Trayectoria: -------------------------
	Evolution<<< grid_size,block_size >>> (d_He, Npart, init, d_Q, d_Q_container);

	// After Evol
	gettimeofday(&finish,NULL);
	elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)/1.0e6;
	printf("Elapsed time AFTER evolution: \t %f sec.\n", elapsed_time);
	

	checkCUDAError("Kernel GPU: failed \n");
	/*  ********   */
	HANDLE_ERROR(cudaMemcpy(&Q, d_Q, NQ*NQ*sizeof(double), cudaMemcpyDeviceToHost));
	checkCUDAError("copy to CPU: failed \n");

	HANDLE_ERROR(cudaMemcpy(&Q_container, d_Q_container, 10*NQ*NQ*sizeof(double), cudaMemcpyDeviceToHost));
	checkCUDAError("copy to CPU: failed \n");
	
	HANDLE_ERROR(cudaMemcpy(&He, d_He, Npart*sizeof(Part), cudaMemcpyDeviceToHost));
	checkCUDAError("copy to CPU: failed \n");

	HANDLE_ERROR(cudaFree(d_He));
	HANDLE_ERROR(cudaFree(d_Q));
	HANDLE_ERROR(cudaFree(d_Q_container));
	

	int jj;

	// After Memcpy
	gettimeofday(&finish,NULL);
	elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)/1.0e6;
	printf("Elapsed time AFTER Memcopy: \t %f sec.\n", elapsed_time);

	// Posiciones finales y estadisticas--------------------
	//fprintf(File_FC,"Número - tiempo - r - theta - z - Vr - Vtheta - Vz - E (kev) - psi - pitch - sentido\n");
	//fprintf(File_Orbit_types, "# Particle Trajectories statistics, pitch=%f, delta=%f\n", pitch_deg, delta);
	fprintf(File_Orbit_types, "# Escapadas\tBananas\tClockwise\tAnticlockwise\tOutliers\n");
	
	int bananas=0; int clockW = 0; int anticlockW = 0; int escapadas = 0; int Outliers = 0;
	int reentrantes = 0;
	
	for(ip=0;ip<Npart;ip++){  
		/*
		// IONIZATION  \\
		// Si no todas las particulas se ionizaron pueden haber problemas por no estar bien alojada la mem de ionization_data
		fprintf(File_Ion," %f \t %f \t %f \t %f \t %f\n", He[ip].Ionization_data[0], He[ip].Ionization_data[1], He[ip].Ionization_data[2],
		He[ip].Ionization_data[3], He[ip].Ionization_data[4]);
		*/

		// ESCAPED PARTICLES --- if escaped, save in file
		if(He[ip].state == 0){
			fprintf(File_Esc," %f \t %f \t %f \t %f \t %f\n", He[ip].Escaped_data[0], He[ip].Escaped_data[1], He[ip].Escaped_data[2], He[ip].Escaped_data[3], He[ip].Escaped_data[4]);
		}

		Part_charge[He[ip].q] = Part_charge[He[ip].q]+1;
		r = sqrt( He[ip].r[0]*He[ip].r[0] + He[ip].r[1]*He[ip].r[1]);
		x = sqrt( (r - hR0)*(r - hR0) + He[ip].r[2]*He[ip].r[2] );
		//printf("state= %e\n", He[ip].state);
		/*
		fprintf(File_FC," %d %f \t  %.5e \t %.5e \t %.5e \t%.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %d  \n",
			//			He[ip].time, rg[0],rg[1],rg[2],
				ip,He[ip].time,He[ip].r[0], He[ip].r[1], He[ip].r[2],
				He[ip].v[0], He[ip].v[1], He[ip].v[2], 
				He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);
		*/
		if(He[ip].flag == 1){
			reentrantes++;
		}

		if(He[ip].state == 0){
			escapadas += 1;
		} else if(He[ip].state == 1){
			bananas += 1;
		} else if(He[ip].state == 2){
			clockW += 1;
		} else if(He[ip].state == 3){
			anticlockW += 1;
		} else if(He[ip].state == 4){
			Outliers += 1;
		} 
	}
	printf("%d, escaparon y volvieron!\n", reentrantes);
	fprintf(File_Orbit_types, "%d\t%d\t%d\t%d\t%d\n", escapadas, bananas, clockW, anticlockW, Outliers);


	// Diagnostics:
	for (int it_diag = 0; it_diag<10; it_diag++){
		for(ip=0;ip<Npart;ip++){  
			fprintf(File_Diag, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", it_diag, He[ip].Diagnosis_data[it_diag][0], 
				He[ip].Diagnosis_data[it_diag][1], He[ip].Diagnosis_data[it_diag][2], He[ip].Diagnosis_data[it_diag][3], 
				He[ip].Diagnosis_data[it_diag][4], He[ip].Diagnosis_data[it_diag][5], He[ip].Diagnosis_data[it_diag][6], 
				He[ip].Diagnosis_data[it_diag][7], He[ip].Diagnosis_data[it_diag][8], He[ip].Diagnosis_data[it_diag][9]);
		}

		// Save each Q as vector
		for (int i=0; i<NQ-1; i++){
			for(int j=0; j<NQ; j++){
				fprintf(File_Edist, "%f\t", Q_container[(NQ*NQ*it_diag)+(i*NQ+j)]);
			}
		}
		for(int j=0; j<NQ-1; j++){
				fprintf(File_Edist, "%f\t", Q_container[(NQ*NQ*it_diag)+(NQ*(NQ-1)+j)]);
			}
		fprintf(File_Edist, "%f\n", Q_container[(NQ*NQ*it_diag)+(NQ*NQ-1)]);
		// Note: Q _might_ be accidentally transposed. Will not fix this because it's a trivial fix in python (Q=Q.T)
	}

	//---------------------------------------

	gettimeofday(&finish,NULL);
	
	/* *** More statistical results ********** */
	fprintf(File_St,"\n Final \n");
	elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)/1.0e6;
	fprintf(File_St,"Elapsed time: \t %f sec.\n", elapsed_time);
	//fprintf(File_Stat,"N process: \t %d \n", Nprocess);
	fprintf(File_St,"N He0: \t %d \n", Part_charge[0]);
	fprintf(File_St,"N He1: \t %d \n", Part_charge[1]);
	fprintf(File_St,"N He2: \t %d \n", Part_charge[2]);
	printf("Elapsed time: \t %f sec.\n", elapsed_time);
	
	fclose(File_IC);
	//fclose(File_FC);
	fclose(File_St);
	fclose(File_Orbit_types);
	//fclose(File_Ion);
	fclose(File_Esc);
	fclose(File_Diag);
	fclose(File_Edist);
	
	

	return 0;
}  /* end main */