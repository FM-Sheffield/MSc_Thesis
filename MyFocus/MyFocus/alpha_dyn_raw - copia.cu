#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include "curso.h"

/* ********** Global parameters. ******************/
// Plasma and reactor parameters
// HOST ******************
const double hB_0   = 2.;				// Campo Magnetico Toroidal en el eje (Tesla)
const double a_cm   = 50.;				// Radio menor del Toroide
const double R_cm   = 171.0;				// Radio mayor del Toroide
const double hR0    = R_cm/a_cm;                        // Radio normalizado
const double Ep_MeV = 0.093;                              // Energía del proyectil (inicial)
const double hmu    = 2.0;                              // fracción masa proyectil/masa proton
//const double hEa    = 0.875*hmu;                        // MeV (E(MeV) = v^2 * Ea)  //Unit of energy
const int    Npart  = 100;                       	      // Numero de partículas
const int    hNstep = 2500;				// Limite paso temporales.
const double hDt    = 0.16;                              // Temporal step
const double hZp    = 1.0;                              // Numero atomico proyectil
double hy;
const double hta    = 1.0439E-8*hmu/(hZp*hB_0);		// sec. (Campo en Teslas) (ITER: 3.94E-9 s)
// constants (Use for adimensionalizations)
const double hc_cgs = 2.9979E10;			// speed of light
const double ha0_cgs = 5.2918E-9;			// Bohr radius
const double hmp_au = 1836.2;				// proton mass (in a.u.)

// DEVICE ***************
__device__ double dEp_MeV= Ep_MeV;
__device__ double E_0    = 0.0;				// Campo electrico de referencia
__device__ double B_0    = hB_0;
__device__ double R0     = R_cm/a_cm;
__device__ double mu_i   = 1.0;				// Fraccion masa ion-target/masa proton.
__device__ double Zb     = 1.0;				// Atomic number of Plasma ions 
__device__ double n_e    = 1.0;			// Densidad e- y p+ del Plasma (CORE) (10E14 cm-3)
__device__ double nH     = 0.01;			// Hydrogen impurities (10E14 cm-3)
__device__ double nH2    = 0.005;			// Molecular H impurity (10E14 cm-3)
__device__ double nHe    = 0.005;			// Idem Helio.
__device__ int    Nstep  = hNstep;			// Limite pasos temporales
__device__ double Zp     = hZp;				// Numero atomico proyectil
__device__ double mu     = hmu;                         // fraccion masa proyectil/masa proton
__device__ double Dt     = hDt;				//
__device__ double da_cm  = a_cm;
__device__ double ta     = hta;
__device__ __constant__ double PI = 3.1415926535897932385;
__device__ double c_cgs = hc_cgs;
__device__ double a0_cgs = ha0_cgs;
__device__ double mp_au = hmp_au;

 struct Part {
	double E_keV; 			// Energia en keV. (los files estan en esta unidad).
	int Z;  			// Numero atomico.
	int q;				// carga neta. (Adimensional)		
	double r[3];			// posición. Coord. cartesianas. (Adimensional)
	double v[3];			// velocidad. Coord. cartesianas. (Adimensional)
	double time;			// time of particle evolution.
	int state;			// 1 = plasma; 0 = vacuum; -1 = wall/divertor. Las coordenadas de flujo para flux<0.01 no estan bien definidas
	int sense;			// sense of rotation
   double pitch;
   double flux;
    int flag;
	};

// Control Trayectoria:
struct Position {
	double r[3];
	double rg[3];
	};
// ----------------

#include "Magnetic_field.h"
#include "General_functions.h" // Utiliza "struct Part" y los #define a_cm, mu, etc!!

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
	B_Asdex(r,z, &B_equilibrio[0], &psi);		 
	//	B_Asdex(B, E,r, qq, z,time,y);
	*s_flux=psi;

	double	Br=B_equilibrio[0];
	double	Bt= B_equilibrio[1];				
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

//Control Trayectoria: ---------------------------
__global__ void Evolution ( struct Part * d_He, int Npart, long init) {

  // -----------------------------------------------
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

	double initial_proyection=0.0,proyection=0.0;
	//Control Trayectoria: -------
	int Ntraj = Nstep/1000;
	int j = 0;

	kk=0;

	if(id < Npart) {
		n = 0; 		
		initial_proyection = Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
		d_He[id].flux=s_flux;
		do{ 
		  	y = 14.457*sqrt(mu*dEp_MeV)/( Zp*B_0*da_cm);

			RK46_NL(d_He+id, y);
			//Boris_c(d_He+id, y);
			n++;
			// ----------------------------------------------------------------

			proyection=Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
			d_He[id].flux=s_flux;
			if(s_flux<0.01)
			  d_He[id].state=0;
			else
			  d_He[id].state=1;

			d_He[id].pitch=proyection;
			if((proyection*initial_proyection)<0.0){ 
			  d_He[id].sense=0;
			  //printf("Pasante= %d\n",atrapada[j]);
			}
		}while(n<Nstep && d_He[id].state>0 );
	}
}


int main(){
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
	FILE *File_IC = fopen("time0000.dat","w");
	if(File_IC == NULL){
		printf("Error File_IC");
		exit(1);}

	FILE *File_FC = fopen("time0001.dat","w");
	if(File_FC == NULL){
		printf("Error File_FC");
		exit(1);}
		
	FILE *File_St = fopen("SR_1MeV0_0x20_He_e_euler.dat","w");
	if(File_St == NULL){
		printf("Error File_St");
		exit(1);}

	 /*********************************************/ 

	/* Random numbers initialization *****/
	double f;
	time_t tran = time(NULL);
	init = labs(init - tran);
	long *ptrinit = &init;
	for(ip=0;ip<1000;ip++)
		f = ran2(ptrinit);		

	/* ***** Particle Initialization *********/

	Init_rv(&xx[0],&yy[0],&zz[0],&vx[0],&vy[0],&vz[0],&tiempo,Npart);

	double hy = 14.457*sqrt(hmu*Ep_MeV)/( hZp*hB_0*a_cm);

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

		He[ip].pitch = Proyection(He[ip].r[0],He[ip].r[2],He[ip].v[0],He[ip].v[1],He[ip].v[2],&s_flux);
		He[ip].flux =s_flux;
		
		if(He[ip].pitch>0)
		  He[ip].sense =1;
		else
		  He[ip].sense =-1;
		
		if(s_flux<0.01)		  
		  He[ip].state = 0;
		else
		  He[ip].state = 1;
		
		He[ip].time = 0.0;

		fprintf(File_IC,"%d %f %f \t %f \t %f \t %f \t %f \t %f %f %f %f %d\n",ip,He[ip].time, He[ip].r[0], He[ip].r[1],He[ip].r[2],
			He[ip].v[0], He[ip].v[1], He[ip].v[2],He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);

	}
	printf("E in: %.14f keV \n", He[0].E_keV);
	printf("x in: %.14f \n", x);

	
	/* ***** Output Statistical */
	fprintf(File_St, "gamma: \t %f \n", 14.457*sqrt(hmu*Ep_MeV)/( hZp*hB_0*a_cm));
	fprintf(File_St,"Initial \n");
	fprintf(File_St, "Dt: \t %f \n", hDt);
	fprintf(File_St, "Nstep: \t %d \n", hNstep);
	fprintf(File_St, "Simul time: \t %f msec. \n", (double)hNstep*hDt*hta*1000.0);
	fprintf(File_St, "Npart: \t %d \n", Npart);
	fprintf(File_St, "Z beam: \t %d \n", (int)hZp);
	fprintf(File_St, "Ep in: \t %f keV \n", Ep_MeV*1000.0);
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
	int numthreads = 32;
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

	for(ip=0;ip<Npart;ip++){
		Part_charge[He[ip].q] = Part_charge[He[ip].q]+1;
		r = sqrt( He[ip].r[0]*He[ip].r[0] + He[ip].r[1]*He[ip].r[1]);
		x = sqrt( (r - hR0)*(r - hR0) + He[ip].r[2]*He[ip].r[2] );

		fprintf(File_FC," %d %f \t  %.5e \t %.5e \t %.5e \t%.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %d  \n",
			//			He[ip].time, rg[0],rg[1],rg[2],
				ip,He[ip].time,He[ip].r[0], He[ip].r[1], He[ip].r[2],
				He[ip].v[0], He[ip].v[1], He[ip].v[2], 
				He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);
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
	fclose(File_FC);
	fclose(File_St);
	
	return 0;
}  /* end main */
