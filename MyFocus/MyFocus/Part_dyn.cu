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


/* ********** Global parameters. ******************/
// Plasma and reactor parameters  (normalized, DIIID)
// HOST ******************
const double hB_0   = 2.2;				// Campo Magnetico Toroidal en el eje (Tesla)
const double a   = 0.67 / 1.67;			// Radio menor del Toroide (previously a_cm)
const double pitch_deg = 60; 			// Pitch en grados, para Init_CI_costado
const int gridsize = 10;					// Gridsize, para Init_CI_costado
const double delta = 0.61;				// equilibrium triangularity
const double R   = 1;				// Radio mayor del Toroide (previously R_cm)
const double hR0    = R;                       // Radio normalizado
const double Ep_MeV = .08;                              // Energía del proyectil (inicial, Mev)
const double hmu    = 2.0;                              // fracción masa proyectil/masa proton (creo)
const int    Npart  = gridsize*gridsize;           // Numero de partículas
const int    hNstep = 56000000;				// Limite paso temporales.
const int m_steps = 10; 					// number of time steps to measure position
const double hDt    = 0.16;                              // Temporal step (normalized)
const double hZp    = 1.0;                              // Numero atomico proyectil
double hgamma;
const double hta    = 1.0439E-8*hmu/(hZp*hB_0);		// sec. (Campo en Teslas) (ITER: 3.94E-9 s) (PREGUNTAR-> el tiempo de ciclotron)
// constants (Used for adimensionalizations)
const double hc_cgs = 2.9979E10;			// speed of light
const double ha0_cgs = 5.2918E-9;			// Bohr radius
const double hmp_au = 1836.2;				// proton mass (in a.u.)

// DEVICE ************************
__device__ double dEp_MeV= Ep_MeV;
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
	};

#include "Magnetic_field.h"
#include "General_functions.h" // Utiliza "struct Part" y los #define a_cm, mu, etc!! -> Por eso el include está luego de struct Part
#include "Elastic_collision_module.h"  // Módulo de colisiones elásticas

// Control Trayectoria:
struct Position {
	double r[3];
	double rg[3];
	};
// ----------------
//da la proyección (v_paralela) y actualiza el valor del flujo. Es un poco confuso
__host__ __device__ double Proyection(double r,double z, double vr,double vt,double vz,double *s_flux){
	//variables para las velocidades
	double v[3]={0.0};
	double psi;
	//Variables para los campos para no calcular repetidamente
	double B_equilibrio[3], modB=0.0;	
	//variables auxiliares
	double proyection=0.0;
	double qq,time,y;
	//empiezo el calculo de las velocidades
	// B_Asdex(r,z, &B_equilibrio[0], &psi);		 
	// B_Asdex(B, E,r, qq, z,time,y);
	B_Analitico(r,z, &B_equilibrio[0], &psi);  // actualiza el valor de B y s_flux con el eq. analitico
	*s_flux=psi;

	double	Br=B_equilibrio[0];
	double	Bt=B_equilibrio[1];				
	double	Bz=B_equilibrio[2];				
		
		modB=sqrt( Br*Br + Bt*Bt + Bz*Bz );
		//versor paralelo a B
		v[0]=Br/modB;
		v[1]=Bt/modB;
		v[2]=Bz/modB;

		proyection = vr*v[0] + vt*v[1] + vz*v[2];
		//printf("proyection= %e\n", proyection);
		return proyection;				
		}

//Control Trayectoria/Evolución temporal: ---------------------------

__global__ void Evolution ( struct Part * d_He, int Npart, long init) {
	//Evolución temporal "normal", asigna los tipos de órbitas en d_He.state

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int Nec;		//steps for elastic collisions
	int Nic = 1000; 	//steps for inelastic collisions
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
	y = sqrt(10*Ep_MeV) * 0.01758437;  // gamma DIIID, 0.0175 es para 100 keV
	double initial_proyection=0.0,proyection=0.0;
	//Control Trayectoria: ------- 
	int j = 0;

	kk=0;
	short unsigned int Period_tol = 40;  // N° de pasos temporales que puede estar fuera del eq (~1 períodos)

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

			// Elastic collisions: ------------------------------------------
			if(d_He[id].E_keV > 700){
				Nec = 500000;
			}else if(d_He[id].E_keV <= 700 && d_He[id].E_keV > 310){
				Nec = 200000;
			}else{
				Nec = 100000;
			}
			if(n % Nec == 0){
				Elastic_collisions(d_He+id, 1.0*Nec*Dt, &i, init, id);
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
			if(s_flux<0){  
				was_outside = true;
				out_counter++;

				is_in = false;
                
				if (out_counter == Period_tol) {
					d_He[id].state=0;  // escapada
                    break;
                }
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

// Evoluciona una particula (con CUDA) y retorna un puntero con las coordenadas en función del tiempo
__global__ void SingleEvol ( struct Part * d_He,  long init, int ip, struct Position * d_R) {
	//Evolución temporal "normal", asigna las orbitas en d_He.state
	int Npart = 1;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int Nec;		//steps for elastic collisions
	int Nic = 1000; 	//steps for inelastic collisions
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
	y = sqrt(10*Ep_MeV) * 0.01758437;  // gamma DIIID, 0.0175 es para 100 keV
	double initial_proyection=0.0,proyection=0.0;
	//Control Trayectoria: ------- 
	int j = 0;

	kk=0;
	// Period tol no debería ser mayor a 6
	short unsigned int Period_tol = 40;  // N° de pasos temporales que puede estar fuera del eq (~1 períodos)

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
				// d_R[m].r[0] = d_He[0].r[0];	d_R[m].r[1] = d_He[0].r[1];	d_R[m].r[2] = d_He[0].r[2];

				// guardando el centro de giro:
				double cg[3];
				centro_giro(d_He+id, cg, y);  
				d_R[m].r[0] = cg[0];	d_R[m].r[1] = cg[1];	d_R[m].r[2] = cg[2];
				m++;
			}
			
			RK46_NL(d_He+id, y);
			//Boris_c(d_He+id, y);
			n++;

			// Elastic collisions: ------------------------------------------
			if(d_He[id].E_keV > 700){
				Nec = 500000;
			}else if(d_He[id].E_keV <= 700 && d_He[id].E_keV > 310){
				Nec = 200000;
			}else{
				Nec = 100000;
			}
			if(n%Nec == 0){
				// printf("Collision! \n");
				Elastic_collisions(d_He+id, 1.0*Nec*Dt, &i, init, id);

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
			if(s_flux<0){  
				was_outside = true;
				out_counter++;

				is_in = false;
                
				if (out_counter == Period_tol) {
					d_He[id].state=0;  // escapada
                    break;
                }
			} else {
				is_in = true;
				out_counter=0;
				if(was_outside && is_in){
					d_He[id].flag = 1;
				}
			}

			if(n>400000 && (proyection*initial_proyection)<0){  // puedo ponerle más condiciones para determinar mejor las órbitas
				d_He[id].state=1;  // banana
				// break;  // puedo comentar el break para ver la órbita completa
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



// Evoluciona una particula (sin CUDA) y genera un file con las coordenadas en función del tiempo (not working)
void singleP_Evol_Not_implemented (struct Part *d_He2, int ip){
		//# no será que hay problemas de nombre????
		FILE *File_orbit = fopen("singleP_Evol.dat","w");  // Creates a File
		if(File_orbit == NULL){
			printf("Error File_orbit");
			exit(1);}  
			
		fprintf(File_orbit,"# Particula N: %d\n",ip);

		int id = 0;
		
		int Nec;		//steps for elastic collisions
		int Nic = 1000; 	//steps for inelastic collisions
		int n = 0;
		int i,kk;	//semilla random numbers
		int q1 = 2;
		double qq00,qq11,omega=16.e-4,tiempo,tiempo0; //flag
		double s_flux;

		double y;  // gamma

		//sense variables ------------
		double t0 = 0.0;
		double vpar, vpar_t;
		double B[3], E[3];
		y = sqrt(10*Ep_MeV) * 0.01758437;  // gamma DIIID, 0.0175 es para 100 keV
		double initial_proyection=0.0,proyection=0.0;

		//Control Trayectoria: ------- 
		int j = 0;
		kk=0;
		short unsigned int Period_tol = 20;  // N° de pasos temporales que puede estar fuera del eq (1.5 períodos)
		int Allowed_Nstep = hNstep;  // en caso de que salga de del equilibrio

		

			n = 0; 		
			initial_proyection = (*d_He2).pitch;  
			//printf("proyection= %e\n", initial_proyection);  //proyection= 9.000147e-01 para debuggear

			short unsigned int out_counter = 0;  // if out_counter = Period_tol -> escapada
			bool is_in = true;
			bool was_outside = false;

			printf("%f \t %f \t %f \n",(*d_He2).r[0], (*d_He2).r[1], (*d_He2).r[2]);
			do{ 
				H_RK46_NL(d_He2, y);
				//Boris_c(d_He+id, y);

				// saves the full trajectory
				/*if(n==0){
					printf("%f \t %f \t %f \n",(*d_He2).r[0], (*d_He2).r[1], (*d_He2).r[2]);
				}
				fprintf(File_orbit,"%f \t %f \t %f \n",(*d_He2).r[0], (*d_He2).r[1], (*d_He2).r[2]);
				*/
				n++;
				// ----------------------------------------------------------------
				// nueva vparalela
				proyection=Proyection((*d_He2).r[0],(*d_He2).r[2],(*d_He2).v[0],(*d_He2).v[1],(*d_He2).v[2],&s_flux);
				/*if(n==3){
					printf("proyection= %e\n", proyection);  // for debug
				}*/
				(*d_He2).flux=s_flux;
				if(s_flux<0){  
					was_outside = true;
					out_counter++;

					if (n<hNstep-Period_tol && is_in){
						Allowed_Nstep = n + Period_tol;
					}

					is_in = false;
					
					if (out_counter == Period_tol) {
						(*d_He2).state=0;  // escapada
						break;
					}
				} else {
					is_in = true;
					out_counter=0;
					if(was_outside && is_in){
						(*d_He2).flag = 1;
					}
				}

				if(n>10000 && (proyection*initial_proyection)<0){  // puedo ponerle más condiciones para determinar mejor las órbitas
					(*d_He2).state=1;  // banana
					break;
				}
				(*d_He2).pitch=proyection;
				
			}while(n<Allowed_Nstep && (*d_He2).state != 0 );

			if ((*d_He2).state == -1){  // sin asignar 
				if((*d_He2).pitch>0){
					(*d_He2).state = 2;  // Clockwise
				} else if((*d_He2).pitch<0){
					(*d_He2).state = 3;  // Anticlockwise
				} else {
					(*d_He2).state = 4;  // Outlier!
				}
			}
		fclose(File_orbit);
}

int main(){
	// Nota: en realidad xx = r, yy=theta, zz=z
    double xx[Npart],yy[Npart],zz[Npart],vx[Npart],vy[Npart],vz[Npart],tiempo,s_flux;
	
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

	FILE *File_St = fopen("SR_1MeV0_0x20_He_e_euler.dat","w");  // stats
	if(File_St == NULL){
		printf("Error File_St");
		exit(1);}

	FILE *File_Orbit_types = fopen("Orbits.dat","w");  // Conteo de órbitas
	if(File_Orbit_types == NULL){
		printf("Error File_Orbit_types");
		exit(1);}

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
	Init_CI_costado(&xx[0],&yy[0],&zz[0],&vx[0],&vy[0],&vz[0], pitch_deg, gridsize, delta);

	double hgamma = sqrt(10 * Ep_MeV) * 0.01758437;

	for(ip=0;ip<Npart;ip++){
		He[ip].E_keV = Ep_MeV*1000.0;
		He[ip].Z = (int)hZp;
		He[ip].q = (int)hZp;

		He[ip].r[0]=xx[ip];
		He[ip].r[1]=yy[ip];
		He[ip].r[2]=zz[ip];
		He[ip].v[0]=vx[ip];
		He[ip].v[1]=vy[ip];
		He[ip].v[2]=vz[ip];

		// v_paralela y flujo 
		He[ip].pitch = Proyection(He[ip].r[0],He[ip].r[2],He[ip].v[0],He[ip].v[1],He[ip].v[2],&s_flux);
		He[ip].flux = s_flux;
		He[ip].flag = 0;
		
		if(He[ip].pitch>0)
		  He[ip].sense =1;
		else
		  He[ip].sense =-1;
		
		if(s_flux<0)		  
		  He[ip].state = 0;
		else
		  He[ip].state = -1;
		
		He[ip].time = 0.0;
		fprintf(File_IC,"Número - tiempo - r - theta - z - Vr - Vtheta - Vz - E (kev) - psi - vparalela - sentido\n");
		fprintf(File_IC,"%d %f %f \t %f \t %f \t %f \t %f \t %f %f %f %f %d\n",ip,He[ip].time, He[ip].r[0], He[ip].r[1],He[ip].r[2],
			He[ip].v[0], He[ip].v[1], He[ip].v[2],He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);

	}

	printf("E in: %.14f keV \n", He[0].E_keV);
	printf("x in: %.14f \n", x);

	
	/* ***** Output Statistical */
	fprintf(File_St, "gamma: \t %f \n", sqrt(10*Ep_MeV) * 0.01758437);
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

	/***** CUDA ******/
	struct Part *d_He;
	HANDLE_ERROR(cudaMalloc( (void**) &d_He, Npart*sizeof(Part) ));
    HANDLE_ERROR(cudaMemcpy( d_He, &He, Npart*sizeof(Part), cudaMemcpyHostToDevice ));
	checkCUDAError("Particle copy: failed \n");
	
    int dev = 0;
        if(cudaGetDevice(&dev)!= cudaSuccess)
                printf("cudaGetDeviceCount FAILED");

        cudaDeviceProp deviceProp;
        //for(int dev = 0; dev < deviceCount; ++dev){
                cudaGetDeviceProperties(&deviceProp,dev);
                printf("\nPlaca %d: %s \n", dev, deviceProp.name);
	int numthreads = 32;  // 32, se puede probar con otras potencias de 2 para optimizar
	int numblocks = (Npart+numthreads-1)/numthreads;
	dim3 block_size(numthreads);
  	dim3 grid_size(numblocks);

	//Control Trayectoria: -------------------------
	Evolution<<< grid_size,block_size >>> (d_He, Npart, init);
	

	checkCUDAError("Kernel GPU: failed \n");
	/*  ********   */
	HANDLE_ERROR(cudaMemcpy(&He, d_He, Npart*sizeof(Part), cudaMemcpyDeviceToHost));
	checkCUDAError("copy to CPU: failed \n");
	HANDLE_ERROR(cudaFree(d_He));

	int jj;

	// Posiciones finales y estadisticas--------------------
	fprintf(File_FC,"Número - tiempo - r - theta - z - Vr - Vtheta - Vz - E (kev) - psi - pitch - sentido\n");
	fprintf(File_Orbit_types, "# Particle Trajectories statistics, pitch=%f, delta=%f\n", pitch_deg, delta);
	fprintf(File_Orbit_types, "# Escapadas\tBananas\tClockwise\tAnticlockwise\tOutliers\n");
	
	int bananas=0; int clockW = 0; int anticlockW = 0; int escapadas = 0; int Outliers = 0;
	int reentrantes = 0;
	
	bool only_oneP = true;
	for(ip=0;ip<Npart;ip++){
		
		if (only_oneP && He[ip].state == 4){
			only_oneP = false;
			printf("Particle state: %d", He[ip].state);
			printf("\nFlag Particle, ip=%d\n", ip);
			// reseteo el estado de la partícula ip
			He[ip].E_keV = Ep_MeV*1000.0;
			He[ip].Z = (int)hZp;
			He[ip].q = (int)hZp;

			He[ip].r[0]=xx[ip];
			He[ip].r[1]=yy[ip];
			He[ip].r[2]=zz[ip];
			He[ip].v[0]=vx[ip];
			He[ip].v[1]=vy[ip];
			He[ip].v[2]=vz[ip];

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
			
			if(s_flux<0){		  
			He[ip].state = 0;}
			else{
			He[ip].state = -1;}
			
			He[ip].time = 0.0;

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

			checkCUDAError("Kernel GPU: failed \n");
			HANDLE_ERROR(cudaFree(D_HE));

			HANDLE_ERROR(cudaMemcpy( &R, d_R, R_size*sizeof(Position), cudaMemcpyDeviceToHost ));
			checkCUDAError("copy to CPU R: failed \n");
			HANDLE_ERROR(cudaFree(d_R));

			FILE *File_orbit = fopen("singleP_Evol.dat","w");  // Creates a File
			if(File_orbit == NULL){
				printf("Error File_orbit");
			exit(1);}  
			
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
	
	return 0;
}  /* end main */
