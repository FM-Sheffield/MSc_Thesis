/*
Dinámica de partículas cargadas en tokamaks con CUDA
Adaptado del código FOCUS -> https://doi.org/10.1016/j.cpc.2018.07.0180010-4655
Facundo Sheffield - 2022
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include "Random123/philox.h"  // random numbers in GPU
#include "Random123/u01.h"  // to get uniform deviates [0,1]


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
const int    Npart  = 100000;  //            // Numero de partículas
const int    hNstep = 24000000;				// Limite paso temporales. 24000000 @ dt=0.16->40ms
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
	int state;				// -1 = sin determinar; 0 = escapada; 1 = banana; 2 = clockwise; 3 = anticlockwise; 4 = outlier
	int sense;				// sense of rotation
    double pitch;  			// Vparalela al campo (V_par/V=cos(pitch))
    double flux;
    int flag; 				// Indica algún flag, en este caso es 1 si salió y volvió a entrar y 0 else

	//double Ionization_data[5];  // saves certain data at ionization. In order, [r, theta, z, v_pll/v, E_kev] 
	double Escaped_data[5];    // saves data (r, th, z, E_kev, time) of a particle when it escapes

	//double Diagnosis_data[10][10];  // saves certain data of particles for further diagnosis at different times. 
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

// Control Trayectoria:
struct Position {
	double r[3];
	double rg[3];
	};
// ----------------

//Control Trayectoria/Evolución temporal: ---------------------------

__global__ void Evolution ( struct Part * d_He, int Npart, long init) {
	//Evolución temporal "normal", asigna los tipos de órbitas en d_He.state

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	int Nec=80000;		// steps for elastic collisions - 80k is okay for E=80keV
	int next_col=Nec;	// total steps for next collision

	int Nic = 2; 	//steps for inelastic collisions
	int n = 0;
	int i,kk;	//semilla random numbers
	int q1 = 2;
	double qq00,qq11,omega=16.e-4,tiempo,tiempo0; //flag
	double s_flux;

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
		
		// ya están en las condiciones iniciales	
		//initial_proyection = Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
		//d_He[id].flux=s_flux;

		initial_proyection = d_He[id].pitch;
		//printf("proyection= %e\n", initial_proyection);  //proyection= 9.000147e-01 para debuggear

		short unsigned int out_counter = 0;  // if out_counter = Period_tol -> escapada
		bool is_in = true;
		bool was_outside = false;


		do{ 

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
				*/

				// __device__ void Elastic_collisions_SV_euler (struct Part *He, double Dt, double * Ran_EC);
				//Elastic_collisions_PE_MC(d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				//Elastic_collisions_euler (d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				// (Notar que para euler necesito 6 nros gaussianos!!!!)
			}

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
	}
}

/*
Copilot test
// This function evolves the dynamical system in time.
// It is called by the main function.
__global__ void Evolve_system(struct Part *d_He, int N, double Dt, double *Ran_EC, int init, int id){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		Evolve_system_kernel(d_He, N, Dt, Ran_EC, init, idx);
	}
}

*/


// Evoluciona una particula (con CUDA) y retorna un puntero con las coordenadas en función del tiempo
__global__ void SingleEvol ( struct Part * d_He,  long init, int ip, struct Position * d_R) {
	//Evolución temporal "normal", asigna las orbitas en d_He.state
	int Npart = 1;
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int Nec=80000;		// steps for elastic collisions - 80k is okay for E=80keV
	int next_col=Nec;	// total steps for next collision
	
	int Nic = 2; 	//steps for inelastic collisions
	int n = 0;
	int i,kk;	//semilla random numbers
	int q1 = 2;
	double qq00,qq11,omega=16.e-4,tiempo,tiempo0; //flag
	double s_flux;

	double y;// gamma

	//sense variables -----------
	double t0 = 0.0;
	double vpar, vpar_t;
	double B[3], E[3];
	//y = sqrt(10*Ep_MeV) * 0.01758437;  // gamma DIIID, 0.0175 es para 100 keV
	y = sqrt(10*Ep_MeV) * 0.019342807;  // gamma DIIID, 0.0193 es para 100 keV @ 2T
	double initial_proyection=0.0,proyection=0.0;
	//Control Trayectoria: ------- 
	int j = 0;

	kk=0;
	// Period tol no debería ser mayor a 6
	short unsigned int Period_tol = 25;  // N° de pasos temporales que puede estar fuera del eq (~1 períodos)

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
		int m = 0;
		// ya están en las condiciones iniciales	
		//initial_proyection = Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
		//d_He[id].flux=s_flux;

		initial_proyection = d_He[id].pitch;
		//printf("proyection= %e\n", initial_proyection);  //proyection= 9.000147e-01 para debuggear

		short unsigned int out_counter = 0;  // if out_counter = Period_tol -> escapada
		bool is_in = true;
		bool was_outside = false;


		do{ 
			if(n % m_steps == 0){
				// guardando la posición (no queda lindo para m_steps > 5)
				d_R[m].r[0] = d_He[0].r[0];	d_R[m].r[1] = d_He[0].r[1];	d_R[m].r[2] = d_He[0].r[2];

				// guardando el centro de giro:
				/*double cg[3];
				centro_giro(d_He+id, cg, y);  
				d_R[m].r[0] = cg[0];	d_R[m].r[1] = cg[1];	d_R[m].r[2] = cg[2];*/
				m++;
			}
			
			RK46_NL(d_He+id, y);
			//Boris_c(d_He+id, y);
			
			// test
			//printf("Hello?\n");
			//if(d_He[id].r[0] < 1 || d_He[id].r[0] > 2){
				//printf("\nE=%f\n Vr=%f \n Vth= %f \n state: %d\n", d_He[id].E_keV, d_He[id].v[0], d_He[id].v[1], d_He[id].state);
			//	break;
			//}

			n++;
			
			/*if(d_He[id].q != 0){
				printf("Ionizada!\n");
			}*/
			
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

			if(n%Nic == 0 && d_He[id].q == 0 && s_flux > 0){  // preguntar césar
				Inelastic_collisions(d_He+id,(double)Dt*Nic*ta, &i, (int)init, id);

				//printf("carga: %d\t pos_r: %f\n", d_He[id].q, d_He[id].r[0]);

				//if(q1 != d_He[id].q){
				//	Nprocess = Nprocess + 1;
				//	q1 = d_He[id].q;
				//}
			}


			// Elastic collisions: ------------------------------------------
			
			// Arreglar esto: Yo quiero que pasen exactamente Nec pasos entre colision y colision, así como está eso no sucede (como max pasa 1*Nec)
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
				Ran_gauss(&Ran_EC[2], 1);  // mean, sd
				i = i + 1;
				c.v[0] = i;		
				i = i + 1;
				c.v[1] = i;	
				p = philox2x32(c, k);
				Ran_EC[4] = (double) u01_open_open_32_53(p[0]);
				Ran_EC[5] = (double) u01_open_open_32_53(p[1]);
				Ran_gauss(&Ran_EC[4], 1);
				*/	

				// __device__ void Elastic_collisions_SV_euler (struct Part *He, double Dt, double * Ran_EC);
				//Elastic_collisions_PE_MC_euler(d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				//Elastic_collisions_SV_euler (d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				//Elastic_collisions_euler (d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				// (Notar que para euler necesito 6 nros gaussianos!!!!)
			}


			// ----------------------------------------------------------------
			// nueva vparalela
			proyection=Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
			/*if(n==3){
				printf("proyection= %e\n", proyection);  // for debug
			}*/
			d_He[id].flux=s_flux;

			if(d_He[id].r[0]<0.8*(R-a) || d_He[id].r[0]>1.2*(R+a) || d_He[id].r[2]<-1.2*Zx || d_He[id].r[2]>1.2*Zx){
				d_He[id].state=0;  // escapada, fuera de la configuración
			}

			if(d_He[id].q != 0){  // no aplica a particulas neutras
				if(s_flux<0){  
					was_outside = true;
					is_in = false;
					
					if (out_counter == Period_tol) {
						d_He[id].state=0;  // escapada
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

				if(n>500000 && (proyection*initial_proyection)<0){  // puedo ponerle más condiciones para determinar mejor las órbitas
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
			d_He[id].state = 4;  // Outlier, energía cinética negativa o NaN
		}
	}
}

int main(){
	// Nota: en realidad xx = r, yy=theta, zz=z
    double Ipos[3], Ivel[3], tiempo, s_flux;
	double xx[Npart],yy[Npart],zz[Npart],vx[Npart],vy[Npart],vz[Npart];
	
	struct Part He[Npart];
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

	FILE *File_FC = fopen("time0001.dat","w");  // Final Conditions
	if(File_FC == NULL){
		printf("Error File_FC");
		exit(1);}

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
	fprintf(File_Esc,"# Escaped particle coordinates and Energy (KeV).\n# R; theta; z; Energy\n");


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
	HANDLE_ERROR(cudaMalloc( (void**) &d_He, Npart*sizeof(Part) ));
    HANDLE_ERROR(cudaMemcpy( d_He, &He, Npart*sizeof(Part), cudaMemcpyHostToDevice));
	checkCUDAError("Particle copy: failed \n");
	
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
	Evolution<<< grid_size,block_size >>> (d_He, Npart, init);

	// After Evol
	gettimeofday(&finish,NULL);
	elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)/1.0e6;
	printf("Elapsed time AFTER evolution: \t %f sec.\n", elapsed_time);
	

	checkCUDAError("Kernel GPU: failed \n");
	/*  ********   */
	
	HANDLE_ERROR(cudaMemcpy(&He, d_He, Npart*sizeof(Part), cudaMemcpyDeviceToHost));
	checkCUDAError("copy to CPU: failed \n");
	HANDLE_ERROR(cudaFree(d_He));
	

	int jj;

	// After Memcpy
	gettimeofday(&finish,NULL);
	elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)/1.0e6;
	printf("Elapsed time AFTER Memcopy: \t %f sec.\n", elapsed_time);

	// Posiciones finales y estadisticas--------------------
	fprintf(File_FC,"Número - tiempo - r - theta - z - Vr - Vtheta - Vz - E (kev) - psi - pitch - sentido\n");
	//fprintf(File_Orbit_types, "# Particle Trajectories statistics, pitch=%f, delta=%f\n", pitch_deg, delta);
	fprintf(File_Orbit_types, "# Escapadas\tBananas\tClockwise\tAnticlockwise\tOutliers\n");
	
	int bananas=0; int clockW = 0; int anticlockW = 0; int escapadas = 0; int Outliers = 0;
	int reentrantes = 0;
	
	bool only_oneP = false;
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

		if (only_oneP && He[ip].state==0){  // DISCONTINUED -> Should update it 
			// This should probably be its own function
			only_oneP = false;
			printf("Particle state: %d", He[ip].state);
			printf("\nFlag Particle, ip=%d\n", ip);
			// reseteo el estado de la partícula ip
			He[ip].E_keV = Ep_MeV*1000.0;
			He[ip].Z = (int)hZp;
			He[ip].q = 0;


			
			double r = R+a;  // radio exterior del toroide (R_out)

			double Ran[2]; // [ran_theta, ran_z]
			// Spatial distribution
			do {
				Ran[0] = ran2(&init);
				Ran[1] = ran2(&init);
				Ran_gauss(&Ran[0], 1);  

				// Reescale to the characteristic sizes:
				Ran[0] = Ran[0]*theta_beam_sd;
				Ran[1] = Ran[1]*z_beam_disp_ang;

			} while (Ran[0]<-4*theta_beam_sd || Ran[0]>4*theta_beam_sd || Ran[1]<-4*z_beam_disp_ang || Ran[1]>4*z_beam_disp_ang);
			//Ran[0]=0.01; Ran[1]=0.0;

			// FALTA CAMBIAR ESTO PARA LAS VELOCIDADES
			// Initial pos
			He[ip].r[0]= r;
			He[ip].r[1]= theta_beam + Ran[0];
			He[ip].r[2]= z_beam + Ran[1];

			r = He[ip].r[0];
			double z = He[ip].r[2];

			// Velocities, tilt angle refers to the one in the X-Y plane
			double vx = -r/sqrt(r*r+z*z)*0.6;  // -Vmod*sin(ang(z, r))*cos(tilt_angle)
			double vy = -r/sqrt(r*r+z*z)*0.8;
			double vz = -z/sqrt(r*r+z*z);  //-Vmod * cos(ang(z, r))

			

			// Initial velocity:
			He[ip].v[0]=vx*cos(He[ip].r[1]) + vy*sin(He[ip].r[1]);
			He[ip].v[1]=-vx*sin(He[ip].r[1]) + vy*cos(He[ip].r[1]);
			He[ip].v[2]=vz;
			

			printf("Coordenada inicial radial r=%f\n", He[ip].r[0]);
			printf("E (keV): %f", He[ip].E_keV);

			// v_paralela y flujo 
			He[ip].pitch = Proyection(He[ip].r[0],He[ip].r[2],He[ip].v[0],He[ip].v[1],He[ip].v[2],&s_flux);
			He[ip].flux = s_flux;
			He[ip].flag = 0;
			
			if(He[ip].pitch>0){
			He[ip].sense = 1;}
			else{
			He[ip].sense = -1;}
			
			He[ip].state = -1;  // indeterminado
			
			He[ip].time = 0.0;

			// Inelastic col:
			#ifdef Z_1			
				He[ip].n = 1;				// quantum number, fundamental state s1
				He[ip].timeAt = 0;         //(IN SEC.) time for atomic de-excitation
			#endif

			int R_size = hNstep/m_steps;
			struct Position R[R_size];
			struct Position *d_R;
			struct Part *D_HE;

			cudaMalloc( (void**) &d_R, R_size*sizeof(Position) );
			cudaMemcpy( d_R, &R, R_size*sizeof(Position), cudaMemcpyHostToDevice );
			checkCUDAError("Trajectory copy: failed \n");
			
			cudaMalloc( (void**) &D_HE, 1*sizeof(Part) );
			cudaMemcpy( D_HE, &He[ip], 1*sizeof(Part), cudaMemcpyHostToDevice );
			checkCUDAError("Particle copy: failed \n");

			int dev = 0;
			if(cudaGetDevice(&dev)!= cudaSuccess)
					printf("cudaGetDeviceCount FAILED");

			cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp,dev);
            printf("\nPlaca %d: %s \n", dev, deviceProp.name);
			int numthreads = 32;
			int numblocks = (Npart+numthreads-1)/numthreads;
			dim3 block_size(numthreads);
			dim3 grid_size(numblocks);

			SingleEvol<<< grid_size,block_size >>>  (D_HE, init, ip, d_R);

			
			HANDLE_ERROR(cudaMemcpy( &R, d_R, R_size*sizeof(Position), cudaMemcpyDeviceToHost ));
			checkCUDAError("copy to CPU R: failed \n");
			HANDLE_ERROR(cudaFree(d_R));


			HANDLE_ERROR(cudaMemcpy(&He[ip], D_HE, 1*sizeof(Part), cudaMemcpyDeviceToHost));
			checkCUDAError("Kernel GPU: failed \n");
			HANDLE_ERROR(cudaFree(D_HE));

			FILE *File_orbit = fopen("singleP_Evol.dat","w");  // Creates a File
			if(File_orbit == NULL){
				printf("Error File_orbit");
			exit(1);}  
			
			printf("E_final (keV): %f\n", He[ip].E_keV);
			for(int t_=0;t_<(R_size);t_++){
				fprintf(File_orbit,"%f \t %f \t %f \n",R[t_].r[0], R[t_].r[1], R[t_].r[2]);
			}
			fclose(File_orbit);
			//D_HE = &He[ip];
			//printf("D_HE.r[0]=%f\n", (*D_HE).r[0]);  // Lo copia bien al puntero
			//Evol_w_coordinates (struct Part * d_He, long init, double &r_cor, double &theta_cor, double &z_cor)
			//singleP_Evol(D_HE, ip);
		}

		Part_charge[He[ip].q] = Part_charge[He[ip].q]+1;
		r = sqrt( He[ip].r[0]*He[ip].r[0] + He[ip].r[1]*He[ip].r[1]);
		x = sqrt( (r - hR0)*(r - hR0) + He[ip].r[2]*He[ip].r[2] );
		//printf("state= %e\n", He[ip].state);

		fprintf(File_FC," %d %f \t  %.5e \t %.5e \t %.5e \t%.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %d  \n",
			//			He[ip].time, rg[0],rg[1],rg[2],
				ip,He[ip].time,He[ip].r[0], He[ip].r[1], He[ip].r[2],
				He[ip].v[0], He[ip].v[1], He[ip].v[2], 
				He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);

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
	fclose(File_FC);
	fclose(File_St);
	fclose(File_Orbit_types);
	//fclose(File_Ion);
	fclose(File_Esc);
	

	return 0;
}  /* end main */
