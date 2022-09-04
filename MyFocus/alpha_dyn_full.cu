#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "Random123/philox.h"
#include "Random123/u01.h"    // to get uniform deviates [0,1]
#include <cuda.h>
#include "curso.h"
/* ******************************************************** */
/*  Para LOS RANDOMS PUEDO PASAR LA SEMILLA POR EL KERNEL!  */
/*  ASI USO EL TIEMPO COMO ALEATORIEDAD */



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

//esto sirve para dimensionar la malla
//const int    N_r  = 401; //malla para la pertubacion B1r1,B1r2,B1z1,B1z2
//const int    N_z  = 401; //malla para la pertubacion B1r1,B1r2,B1z1,B1z2
/* const double    r0  = 1.1; */
/* const double    r1  = 2.2; */
/* const double    z0  = -0.9; */
/* const double    z1  = 0.9; */



/* __device__ int    dN_r  = N_r; //malla para la pertubacion B1r1,B1r2,B1z1,B1z2 */
/* __device__ int    dN_z  = N_z; //malla para la pertubacion B1r1,B1r2,B1z1,B1z2 */
/* //limites para la perturbacion */
/* __device__ double    dr0  = r0; */
/* __device__ double    dr1  = r1; */
/* __device__ double    dz0  = z0; */
/* __device__ double    dz1  = z1; */


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
//__device__ int    Nic     = 5;			// Cada cuánto pregunto por procesos atómicos inelasticos.
__device__ int    Nec	 = 100;			// Cada cuanto computo las colisiones elásticas.
__device__ double Zp     = hZp;				// Numero atomico proyectil
__device__ double mu     = hmu;                         // fraccion masa proyectil/masa proton
__device__ double Dt     = hDt;				//
__device__ double da_cm  = a_cm;
//__device__ double Ea     = hEa;
__device__ double ta     = hta;
__device__ __constant__ double PI = 3.1415926535897932385;
__device__ double c_cgs = hc_cgs;
__device__ double a0_cgs = ha0_cgs;
__device__ double mp_au = hmp_au;



// Especificar que está todo adimensionalizado!!!
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
   /* double resr[200]; //puntos de resonancia */
   /* double resq[200]; */
   /* double resz[200]; */
   
   //   double qq0;
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

//#include "Inelastic_collision_module.h"
#include "Elastic_collision_module.h"

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
//__global__ void Evolution ( struct Part * d_He, struct Position * d_R, int Npart, long init,double *B1r1,double *B1r2,double *B1z1,double *B1z2) {
//__global__ void Evolution ( struct Part * d_He, int Npart, long init,double *B1r1,double *B1r2,double *B1z1,double *B1z2) {
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
	//double r = sqrt(d_He[id].r[0]*d_He[id].r[0] + d_He[id].r[1]*d_He[id].r[1]);
	//	double r = d_He[id].r[0];
	//	double x = sqrt( (r-R0)*(r-R0) + d_He[id].r[2]*d_He[id].r[2] );

	double y;// gamma

	//sense variables -----------
	//#ifdef MODE_INIT
	double t0 = 0.0;
	double vpar, vpar_t;
	double B[3], E[3];

	double initial_proyection=0.0,proyection=0.0;
	// Descomentar
	// Random numbers C :
	// Random numbers initialization -------------
	/* philox2x32_ctr_t   c={{}}; */
	/* philox2x32_ukey_t  uk = {{}}; */
	/* uk.v[0] = id + (int)init;		 */
	/* philox2x32_key_t   k = philox2x32keyinit(uk); */
	/* philox2x32_ctr_t   p;	 */
	/* double Ran1,Ran2; */
	/* double Ran_EC[4];  // pitch-energy */
	//double Ran_EC[6];  // euler	

	//Control Trayectoria: -------
	int Ntraj = Nstep/1000;
	int j = 0;
	// --------------------

	//	printf("gamma %lf \n",14.457*sqrt(mu*dEp_MeV)/( Zp*B_0*da_cm));

	// Descomentar
	/* for(i=0;i<1000;i++){ */
	/* 	c.v[0] = i; */
	/* 	p = philox2x32(c, k); */
	/* } */

	kk=0;

	if(id < Npart) {
		// Podria incorporar la inicializacion aca.
		// Pero habria que ver si quiero imprimir las posiciones iniciales.
		n = 0; 
		/* tiempo0=0.0; */
		/* qq00=d_He[id].r[1]; */
		/* tiempo=d_He[id].time; */
		/* d_He[id].flag=0; */
		
		initial_proyection 	=Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
		d_He[id].flux=s_flux;
		do{ 
		  	y = 14.457*sqrt(mu*dEp_MeV)/( Zp*B_0*da_cm);
		  //hugo gamma=
			//y=0.048669;
			// Control trayectoria: ----------------------

			//hugo, aca uso rg para copiar la velocidad
			/* if(n%Ntraj == 0){ */
			/*   //if(n>=4800000 && n%200==0){ */
			/*   	d_R[j].r[0] = d_He[0].r[0];	d_R[j].r[1] = d_He[0].r[1];	d_R[j].r[2] = d_He[0].r[2]; */
			/* 	//	centro_giro(d_He, &d_R[j].rg[0], y); */
			/* 		d_R[j].rg[0] = d_He[0].v[0];	d_R[j].rg[1] = d_He[0].v[1];	d_R[j].rg[2] = d_He[0].v[2]; */

			/* 	j++; */
			/* } */

			
			

			// -------------------------------------------
			//	RK46_NL(d_He+id, y,B1r1,B1r2,B1z1,B1z2);
	RK46_NL(d_He+id, y);

			//		Boris_c(d_He+id, y);
			n++;

			// Inelastic collisions:
		/*	if(n%Nic == 0){
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
			// Elastic collisions: ------------------------------------------
			if(d_He[id].E_keV > 700){
				Nec = 500000;
			}else if(d_He[id].E_keV <= 700 && d_He[id].E_keV > 310){
				Nec = 200000;
			}else{
				Nec = 100000;
			}
			if(n%Nec == 0){
				//Elastic_collisions(d_He+id, uk, (double)Nec*Dt, &i);
				// Randoms numbers needed in elast. collision -------
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

				// __device__ void Elastic_collisions_SV_euler (struct Part *He, double Dt, double * Ran_EC);
				Elastic_collisions_PE_MC(d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				//Elastic_collisions_euler (d_He+id,(double)Nec*Dt, &Ran_EC[0]);
				// (Notar que para euler necesito 6 nros gaussianos!!!!)
			}
			// -------------------------------------------------------------------------------
			//control(1)
			//d_He[id].r[0] = Ran_EC[0]; d_He[id].r[1] = Ran_EC[1]; d_He[id].r[2] = Ran_EC[2];
			//d_He[id].v[0] = Ran_EC[3]; d_He[id].v[1] = Ran_EC[4]; d_He[id].v[2] = Ran_EC[5];  
			//d_Datos[n] = Ran1;

			//n++;
			//	r = sqrt(d_He[id].r[0]*d_He[id].r[0] + d_He[id].r[1]*d_He[id].r[1]);

			//			r=d_He[id].r[0];

			//			x = sqrt( (r-R0)*(r-R0) + d_He[id].r[2]*d_He[id].r[2] );
			// ATENCION!!!! Para B constante suprimir la condicion sobre x!!!!
			//x = 1.0;
			//		}while(n<Nstep && (d_He[id].q > 0) && x < 1.001);


			proyection=Proyection(d_He[id].r[0],d_He[id].r[2],d_He[id].v[0],d_He[id].v[1],d_He[id].v[2],&s_flux);
			d_He[id].flux=s_flux;
			if(s_flux<0.01)
			  d_He[id].state=0;
			else
			  d_He[id].state=1;

			d_He[id].pitch=proyection;
			if((proyection*initial_proyection)<0.0){ 
			  d_He[id].sense =0;
			  //printf("Pasante= %d\n",atrapada[j]);
			}
			
		}while(n<Nstep && d_He[id].state>0 );
		
		//__syncthreads();
		//control
		//d_He[id].r[0]=CE_He2_H[10][1];
		//d_He[id].r[1]=Ran2;
	}
}









int main(){
  double xx[Npart],yy[Npart],zz[Npart],vx[Npart],vy[Npart],vz[Npart],tiempo,s_flux;
	//double Datos[hNstep];
	//double *d_Datos;

	/* **** Adimensionalization units *******************************/
	
	/* Unidad de campo: 		B_0  		En Teslas (1E4 Gauss).
	 * Unidad de longitud: 		a_cm		(declarada con #define)
	 * Unidad de carga: 		q_e 		(Carga electron - no hace falta declarar).
	 * Unidad de velocidad: 	v_a			(velocidad inicial de las particulas)
	 * Unidad de tiempo: 		t_a			Inversa de la frecuencia de ciclotrón
	 */ 
	
	
	struct Part He[Npart];
	int ip;         			// Particle index
	double x,r;				// Initial x position (initializated in fn init_r)
	double rg[3];				// Guiding center.
	//double ptheta;				// canonic momentum.

	/* //Control Trayectoria: */
	/* struct Position R[1000]; */
	/* x = 0.0; //la defino pero no se usa. */
	// --------------------

	int Part_charge[3]={0,0,0};		// Final charge state counter.
	//int Nprocess = 0;				// Number of ocurred processes.
	//int q1 = 2;						// aux. variable.
	
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
	/* //Control Trayectoria: ----- 	 */
	/* FILE *File_Traj = fopen("rk4_gpu.dat","w"); */
	/* if(File_Traj == NULL){ */
	/* 	printf("Error File_Traj"); */
	/* 	exit(1);} */

	/* //poincare cinetico */
	/* 	FILE *File_poin = fopen("poinc.dat","w"); */
	/* if(File_Traj == NULL){ */
	/* 	printf("Error File_poin"); */
	/* 	exit(1);} */



	// -------------------------
	 /*********************************************/ 

	/* Random numbers initialization *****/
	double f;
	time_t tran = time(NULL);
	init = labs(init - tran);
	long *ptrinit = &init;
	for(ip=0;ip<1000;ip++)
		f = ran2(ptrinit);		
	/**inicializar el campo perturbado!!!**/

	//	FILE *file_in_CamposMagneticos= fopen("CamposMagneticoPerturbado_A6_n401.txt", "r");


		//	assert(file_in_CamposMagneticos!=NULL);	


	/* double *h_B1r1 = (double *)malloc(N_r*N_z*sizeof(double)); */
	/* double *h_B1z1 = (double *)malloc(N_r*N_z*sizeof(double)); */
	/* double *h_B1r2 = (double *)malloc(N_r*N_z*sizeof(double)); */
	/* double *h_B1z2 = (double *)malloc(N_r*N_z*sizeof(double)); */
	/* int i,j; */
	/* double aux; */
	/* for(i=0;i<N_z;i++){ */
	/*   for(j=0;j<N_r;j++){ */
	/*     fscanf(file_in_CamposMagneticos,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&aux,&aux,&h_B1r1[i*N_r+j],&h_B1r2[i*N_r+j],&h_B1z1[i*N_r+j],&h_B1z2[i*N_r+j]);  */
	/*     // fscanf(in,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&aux,&aux,&B1r1[i][j],&B1r2[i][j],&B1z1[i][j],&B1z2[i][j]);  */
	/* 	} */
	/* } */
		
	/* fclose(file_in_CamposMagneticos);	 */

	/* double *d_B1r1; */
	/* double *d_B1z1; */
	/* double *d_B1r2; */
	/* double *d_B1z2; */

	/* HANDLE_ERROR(cudaMalloc((void**)&d_B1r1,  N_r*N_z*sizeof(double))); */
	/* HANDLE_ERROR(cudaMalloc((void**)&d_B1z1,  N_r*N_z*sizeof(double))); */
       	/* HANDLE_ERROR(cudaMalloc((void**)&d_B1r2,  N_r*N_z*sizeof(double))); */
	/* HANDLE_ERROR(cudaMalloc((void**)&d_B1z2,  N_r*N_z*sizeof(double))); */

	/* HANDLE_ERROR(cudaMemcpy(d_B1r1,h_B1r1, N_r*N_z*sizeof(double),cudaMemcpyHostToDevice)); */
	/* HANDLE_ERROR(cudaMemcpy(d_B1z1,h_B1z1, N_r*N_z*sizeof(double),cudaMemcpyHostToDevice)); */
       	/* HANDLE_ERROR(cudaMemcpy(d_B1r2,h_B1r2, N_r*N_z*sizeof(double),cudaMemcpyHostToDevice)); */
      	/* HANDLE_ERROR( cudaMemcpy(d_B1z2,h_B1z2, N_r*N_z*sizeof(double),cudaMemcpyHostToDevice)); */
	/* printf("Campo magnetico cargado.\n"); */
	/* checkCUDAError("Pertubartion initialization failed: failed \n"); */
	
	/* ***** Particle Initialization *********/

	Init_rv(&xx[0],&yy[0],&zz[0],&vx[0],&vy[0],&vz[0],&tiempo,Npart);

	//double Ep = Ep_MeV/hEa;
	//double vp = sqrt(Ep);
	double hy = 14.457*sqrt(hmu*Ep_MeV)/( hZp*hB_0*a_cm);
	//hugo gamma
	//double hy=0.048669;
	for(ip=0;ip<Npart;ip++){
		He[ip].E_keV = Ep_MeV*1000.0;
		He[ip].Z = (int)hZp;
		He[ip].q = (int)hZp;
		//		Init_r(&He[ip].r[0]	, &x);
		//	Init_v(&He[ip].v[0]);
		//		Init_gc(&He[ip], hy);

		He[ip].r[0]=xx[ip];
		He[ip].r[1]=yy[ip];
		He[ip].r[2]=zz[ip];
		He[ip].v[0]=vx[ip];
		He[ip].v[1]=vy[ip];
		He[ip].v[2]=vz[ip];

		/* He[ip].r[0] = hR0 + (ip+1)*0.01; */
		/* He[ip].r[1] = 0.0; */
		/* He[ip].r[2] = 0.0; */

		/* Init_v(&He[ip].v[0],&He[ip].r[0]); */

		/* He[ip].v[0] = 0.562703; */
		/* He[ip].v[1] = 0.2; */
		/* He[ip].v[2] = 0.8021; */
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
		//centro_giro(&He[ip],&rg[0],hy);
		/*
			fprintf(File_IC,"%f \t %f \t %f \t %f \t %f \t %f \n", rg[0], rg[1], rg[2], 
			He[ip].v[0], He[ip].v[1], He[ip].v[2]);*/
		fprintf(File_IC,"%d %f %f \t %f \t %f \t %f \t %f \t %f %f %f %f %d\n",ip,He[ip].time, He[ip].r[0], He[ip].r[1],He[ip].r[2],
			He[ip].v[0], He[ip].v[1], He[ip].v[2],He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);

	}
	//printf("vel in: %.14f \n", vp);
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
	
	/* //Control Trayectoria: --------------------------------- */
	/* struct Position *d_R; */
	/* 	cudaMalloc( (void**) &d_R, 1000*sizeof(Position) ); */
	/*  cudaMemcpy( d_R, &R, 1000*sizeof(Position), cudaMemcpyHostToDevice ); */
	/* // --------------------------------------------- */
	/* checkCUDAError("Trajectory copy: failed \n"); */

	/*hugo comento esto
	// Matrices de procesos atomicos ---------
	init_matrices();		//Load reaction rates matrices in Host.
        cudaMemcpyToSymbol( CE_He2_H, hCE_He2_H, 51*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol( CE_He2_H2, hCE_He2_H2, 50*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol( CE_He2_He, hCE_He2_He, 100*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol( CE_He1_H, hCE_He1_H, 51*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol( CE_He1_H2, hCE_He1_H2, 51*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol( CE_He1_He, hCE_He1_He, 101*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol( EII1, hEII1, 35*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol(PII_CE_He1, hPII_CE_He1, 47*6*sizeof(double), size_t(0), cudaMemcpyHostToDevice );

	//Haz de deuterio
	cudaMemcpyToSymbol( ER_D0_D1, hER_D0_D1, 17*6*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( EII_D, hEII_D, 16*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice );
	// ------------------------------------
	hugo*/
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
	//	Evolution<<< grid_size,block_size >>> (d_He, d_R, Npart, init,d_B1r1,d_B1r2,d_B1z1,d_B1z2);
	// -------------------------------------

	//	Evolution<<< grid_size,block_size >>> (d_He, Npart, init,d_B1r1,d_B1r2,d_B1z1,d_B1z2);

	Evolution<<< grid_size,block_size >>> (d_He, Npart, init);
	
	checkCUDAError("Kernel GPU: failed \n");
/*  ********   */
	HANDLE_ERROR(cudaMemcpy(&He, d_He, Npart*sizeof(Part), cudaMemcpyDeviceToHost));
	checkCUDAError("copy to CPU: failed \n");

	HANDLE_ERROR(cudaFree(d_He));

	//Control Trayectoria: -------------------------------------
	//	HANDLE_ERROR(	cudaMemcpy( &R, d_R, 1000*sizeof(Position), cudaMemcpyDeviceToHost ));
	//	checkCUDAError("copy to CPU R: failed \n");
	//	HANDLE_ERROR(cudaFree(d_R));
	/* //hugo, ojo que aca en rg puse la velocidad  */
	/* for(ip = 0; ip<1000;ip++){ */
	/* 	fprintf(File_Traj,"%d \t %.5e \t %.5e \t %.5e \t%.5e \t %.5e \t %.5e \n", */
	/* 		ip,             R[ip].r[0],R[ip].r[1],R[ip].r[2], R[ip].rg[0],R[ip].rg[1],R[ip].rg[2]); */


	/*  }  */

int jj;

 /* double rr,rz,xx,fi,fi2; */

 /* 	for(ip = 0; ip<Npart;ip++) */
 /* 	  { */
 /* 	    fprintf(File_poin,"# %d  %d \n",ip,He[ip].flag); */
 /* 	    for(jj=0;jj<He[ip].flag;jj++) */
 /* 	      { */
 /* 		rr = He[ip].resr[jj]; */
 /* 		rz=He[ip].resz[jj]; */
 /* 		x = sqrt( (rr - hR0-0.1063)*(rr - hR0-0.1063) + rz*rz ); */
 /* 		xx=sqrt( (rr - hR0)*(rr-hR0) + rz*rz ); */
		
 /* 		fi=asin(rz/x);             // !poloidal angle */
 /* 		if(rr < hR0+0.1063)  */
 /* 		  fi=3.14159265359-fi; */
 
 /* 		fi2=asin(rz/xx);             //	!poloidal angle */
 /* 		if(rr < hR0)  */
 /* 		  fi2=3.14159265359-fi2; */

 /* 		fprintf(File_poin,"%d %d %e %e %e %e %e %e %e\n",ip,jj,x,fi,fi2,xx,He[ip].resr[jj],He[ip].resq[jj],He[ip].resz[jj]); */
 /* 	      }  */
 /* 	  } */
	// ---------------------------------------------------


	// Posiciones finales y estadisticas--------------------



	for(ip=0;ip<Npart;ip++){
		Part_charge[He[ip].q] = Part_charge[He[ip].q]+1;
		r = sqrt( He[ip].r[0]*He[ip].r[0] + He[ip].r[1]*He[ip].r[1]);
		x = sqrt( (r - hR0)*(r - hR0) + He[ip].r[2]*He[ip].r[2] );

		//		centro_giro(&He[ip],&rg[0],hy);

		fprintf(File_FC," %d %f \t  %.5e \t %.5e \t %.5e \t%.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %d  \n",
			//			He[ip].time, rg[0],rg[1],rg[2],
			ip,He[ip].time,He[ip].r[0], He[ip].r[1], He[ip].r[2],
	
				He[ip].v[0], He[ip].v[1], He[ip].v[2], 
				He[ip].E_keV,He[ip].flux,He[ip].pitch, He[ip].sense);
		


		//control:
		//printf("%.5e \t %.5e \t %.5e \n", He[ip].r[0],He[ip].r[1],He[ip].r[2]);
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
	
	
	/*
	// Control en pantalla ----------------------
	//ptheta = He[0].r[0]*He[0].v[1] + Psi(He[0].r[0],He[0].r[2])/(2.0*y);
	
	
	//printf("Ptheta final: %.14f \n", ptheta);
	printf("Tiempo simulado: %.14f sec.\n", He[0].time*hta);
	*/
	printf("Elapsed time: \t %f sec.\n", elapsed_time);
	
	//Control trayectoria ----------
	//fclose(File_Traj);
	//	fclose(File_poin);
	// -----------------------------
	fclose(File_IC);
	fclose(File_FC);
	fclose(File_St);
	
	return 0;
}  /* end main */




