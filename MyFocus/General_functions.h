/* *********************************************
 ********* CATALOGO DE FUNCIONES ***************
 ********************************************* */
		

long init=2147483647; // Primo. nro de Mersenne.	


/* *********Particle Evolution ***************/
// Initialization


//void campo_mag(double *B, double r, double qq, double zc);


//void Init_v( double *v,double *r);
//void Init_r(double *r);

void Init_rv(double *xx,double *yy, double *zz, double *vx,double *vy,double *vz,double *tiempo,int npart);
void Init_Neutral_Beam(struct Part *He, double pos[3], double vel[3], double Energy_MeV);


// Boris algorithm:
//__device__ void Boris_c(struct Part *He,double y);
// RK4


//__device__ void RK46_NL(struct Part *He,double y,double *B1r1,double *B1z1,double *B1z2,double *B1r2);
//__device__ void RHS_cil(double *FF, double *u, double y, double q_Z, double time,double *B1r1,double *B1z1,double *B1z2,double *B1r2);


//const double Dt = 0.16;
//double dEp_MeV = 0.08;
__host__ __device__ double Proyection(double r,double z, double vr,double vt,double vz,double *s_flux);

__device__ void RK46_NL(struct Part *He,double y);
__device__ void RHS_cil(double *FF, double *u, double y, double q_Z, double time);
__host__ void H_RK46_NL(struct Part *He,double y);
__host__ void H_RHS_cil(double *FF, double *u, double y, double q_Z, double time);

// Guiding center:
__host__ __device__ void centro_giro(struct Part *He, double *rg, double y);


/* **********plasma profiles ***************/

__device__ double n_ei(double r, double th, double z){     // Density profile, note that n_ei = ne = ni and n_tot = 2n_ei
	double n = 0.5;  // density profile in units of 10E20 m^{-3}  (10^14cm^-3)
	//printf("\nEntró a n_ei\n");
	//printf("n_ei=%f\n", n);
	return n; 
}
__device__ double f_ei(double r); 		// electrones e iones
__device__ double Te(double r, double th, double z){	   // Temperature profile in keV, note that T_tot = 2Te with Te = Ti
	double T_min_keV = 0.05; // We set 0.05 keV as the min temperature in order to avoid numerical issues 
	double psi = Psi_analitico(r, z);
	if (psi > 0){
		//printf("psi=%f\n", psi);
		double p_adim = -S1 * psi * exp(M * M * r * r);
		double T_keV = 24.8341698*p_adim*hB_0*hB_0/(n_ei(r, th, z));  // temperature in KeV
		return T_keV+T_min_keV;; 
	} else {
		return T_min_keV;
	}
}


__device__ double f_n(double r); 		// neutros

 
double ran2(long *idum);			// Num. recipes
__host__ __device__ void Ran_gauss(double *RG, double sigma);
void CrossProduct(double v_A[3], double v_B[3], double c_P[3]);





/*
void  campo_mag(double *B, double r, double qq, double zc){
// Coincide con el de Ricardo.
	// Necesito coord. toroidales:
	
	
  double fi;
  double A=2.624;
	double ep = 1.0/A;
	double R0 = A;
	//	double r = sqrt(xc*xc +yc*yc);
	// Usamos x para la coord toroidal.
	double xx = sqrt((r-R0)*(r-R0) + zc*zc);
	double cfi = (r-R0)/xx;
	double sfi = zc/xx;
	
	//	int mm=1,nn=1;
	
	
	// constantes:
	double ak = 2.4048255577;		// 1er cero de la J_0.
	double p1 = 0.05;			// Ref: R. Farengo, Plasma Phys. Control. Fusion 54 (2012) 025007.
	double I0 = 1.0;
	double alpha = 4.0*p1/(ep*ep);
	double bw = 0.17;					// Campo poloidal en x = 1, theta = 0. Ref: Idem p1.
	double i1 = sqrt(0.25*ep*ep*ak*ak-p1);	// Corriente orden 1.
	double  j1k=j1(ak);
	// calculo la constante c:
	//	double aux = -0.5*(ep/(1.0+ep))*k*j1(k)*(1.0 + 0.5*ep*(1.0+2.0*af/(k*k)));
	//	double C = Bw/aux;		//corroborado con el codigo de Ricardo.
	double  C=-2.0*(1.0+ep)*bw/(ep*j1k*ak*(1.0
					 +ep*(1.0+2.0*alpha/(ak*ak))/2.0));
	 
	 

	
	double bfi;
	
	double bmod2,dbfidx;
	//	   xx=dsqrt((r-A)**2+z**2) !"radial" toroidal coordinate
	fi=asin(zc/xx);             // !poloidal angle
     if(r < A) 
	  fi=3.14159265359-fi;


	double xd=1.0-xx*xx; 
        double j0x=j0(ak*xx);
        double j1x=j1(ak*xx);
        double j0p=-j1x;
        double j1p=j0x-j1x/(ak*xx);

	double den=2.0*(1.0+ep*xx*cfi);
        double dp=1.0+ep*ep*xx*xx;

        double psi0=C*j0x;

       


	double aux=ak*xx+alpha*xx/ak+alpha/(ak*xx);
	

	bfi=-0.5*ep*C*ak*j1x;
	dbfidx=0.5*ep*C*ak*ak*(j1x/(ak*xx)-j0x);
       
        

	
	double psi1=cfi*C*0.5*(xx*j0x+alpha*j1x*xd/ak);

         double  br=-ep*C*(-ak*j1x*sfi+0.5*ep*sfi*cfi*(j1x*(-ak*xx
							    -2.0*alpha/(ak*xx))+j0x*alpha*xd))/den;
           
         double  bz=ep*C*(-cfi*ak*j1x+0.5*ep*(j0x*(1.0+alpha*xd*cfi*cfi)
		  +j1x*(alpha*xd*sfi*sfi/(ak*xx)
			-cfi*cfi*(ak*xx+alpha*(xx+1.0/xx)/ak))))/den;

	 double psi=psi0+ep*psi1;
	 

	  double  bq=2.0*sqrt(1.0+i1*i1*psi*psi)/den;
	  

	  //	 bqp is used to calcualte the perturbed fields. Set bqp=1.0d0 for "cylindrical case"

          B[0]=br;
          B[2]=bz;
          B[1]=bq;
}
*/

/* ********************************************
 ********* CODIGO DE FUNCIONES ****************
 ******************************************** */


// Inicializaciones 

void Init_Neutral_Beam(struct Part *He, double theta_mean, double theta_sd, double z_mean, double z_sd, double Energy_MeV){
	// Inicializa un haz de Deuterio neutro inyecado desde pos
	double s_flux;  // dummy flux variable

	for (unsigned int i = 0; i < Npart; i++){
		He[i].E_keV = Energy_MeV*1000.0;
		He[i].Z = (int)hZp;  // variable global
		He[i].q = 0;  // haz neutro

		double r = 0.965*(R+a);  // radio exterior del toroide (R_out)
		double z = z_mean;
		double theta = theta_mean;

		double Ran[2]; // [ran_theta, ran_z]
		

		// Spatial distribution
		do {
			Ran[0] = ran2(&init);
			Ran[1] = ran2(&init);
			Ran_gauss(&Ran[0], 1);  

			// Reescale to the characteristic sizes:
			Ran[0] = Ran[0]*theta_sd;
			Ran[1] = Ran[1]*z_sd;

		} while (Ran[0]<-4*theta_sd || Ran[0]>4*theta_sd || Ran[1]<-4*z_sd || Ran[1]>4*z_sd);
		//Ran[0]=0.01; Ran[1]=0.0;
		
		// Initial pos
		He[i].r[0]= r;
		He[i].r[1]= theta + Ran[0];
		He[i].r[2]= z + Ran[1];
		// printf("z=%f\ttheta=%f\t", He[i].r[2], He[i].r[1]);
		// redefino r y z:
		r = He[i].r[0];
		z = He[i].r[2];

		
		// Velocities, tilt angle refers to the one in the X-Y plane
		// 0.6 and 0.8 to model the DIII-D data
		double vx = -r/sqrt(r*r+z*z)*0.6;  // -Vmod*sin(ang(z, r))*cos(tilt_angle)
		double vy = r/sqrt(r*r+z*z)*0.8;
		double vz = -z/sqrt(r*r+z*z);  //-Vmod * cos(ang(z, r))
		// printf("z=%f\tvz=%fvx=%f\tvy=%f\ttheta=%f\tr=%f\n", z, vz, vx, vy, theta, r);

		// Initial velocity:
		He[i].v[0]=vx*cos(He[i].r[1]) + vy*sin(He[i].r[1]);
		He[i].v[1]=-vx*sin(He[i].r[1]) + vy*cos(He[i].r[1]);
		He[i].v[2]=vz;

		// v_paralela y flujo 
		He[i].pitch = Proyection(He[i].r[0],He[i].r[2],He[i].v[0],He[i].v[1],He[i].v[2],&s_flux);
		He[i].flux = s_flux;
		He[i].flag = 0;

		if(He[i].pitch>0)
		  He[i].sense =1;
		else
		  He[i].sense =-1;
		
		He[i].state = -1;  // no determinado

		He[i].time = 0.0;

		// Inelastic col:
		#ifdef Z_1			
			He[i].n = 1;				// quantum number, fundamental state s1
			He[i].timeAt = 0;      //(IN SEC.) time in AU, for atomic de-excitation
		#endif
	}
}

void Init_CI_costado(double *r,double *theta, double *z, double *vr,double *vtheta,double *vz, double pitch_deg, unsigned int IC_gridsize, double triangularity){
	// Inicializa IC_gridsize**2 partículas a un costado de la sección transversal con un dado pitch
	double PI = 3.1415926535897932385;
	double X[IC_gridsize]; 
	double Y[IC_gridsize];
	// Crea el grid
	if (triangularity<0){
		for (unsigned int i = 0; i < IC_gridsize; i++) {
			X[i] = 1.125 + i * 0.25 / IC_gridsize;
			Y[i] = 0.1 - i * 0.2 / IC_gridsize;
		}
	} else {
		for (unsigned int i = 0; i < IC_gridsize; i++) {
			X[i] = 1.11 + i * 0.25 / IC_gridsize;
			Y[i] = 0.1 - i * 0.2 / IC_gridsize;
		}
	}

	// Main Loop
	for (unsigned int j = 0; j < IC_gridsize; j++){
		for(unsigned int i = 0; i < IC_gridsize; i++){
			r[IC_gridsize*j+i]=X[i]; theta[IC_gridsize*j+i] = 0; z[IC_gridsize*j+i] = Y[j];

			// Busco v_perpendicular random con ese pitch
			double V_par_abs = cos(PI * pitch_deg / 180.0); // normalizada a v0
			double B[3];
			double E[3];
			magnetic_field(&B[0],&E[0], X[i],0,Y[j]);
			double modB=sqrt( B[0]*B[0] + B[1]*B[1] + B[2]*B[2] );
			double B_versor[3] = { B[0] / modB, B[1] / modB, B[2] / modB };
			double V_par[3] = { V_par_abs * B_versor[0], V_par_abs * B_versor[1], V_par_abs * B_versor[2] };

			// Using ran2:
			long init=2147483647; // Primo, nro de Mersenne.	
            double f;
            time_t tran = time(NULL);
            init = labs(init - tran);
			//init = labs(init - 10);  //same velocity for each particle on every iteration
            long *ptrinit = &init;
            double phi = 2*PI*ran2(ptrinit);
            double th = 2*PI*ran2(ptrinit); 

			double V_per_abs = sin(PI * pitch_deg / 180.0);
			// random angles in the velocity space
            double v_aux[3] = { sin(th) * cos(phi), sin(th) * sin(phi), cos(th) };  // Vraux, Vthetaaux, Vzaux
            double V_per[3] = { 0, 0, 0 };
			CrossProduct(v_aux, B_versor, V_per);
			double v_per_aux_abs = sqrt((V_per[0] * V_per[0] + V_per[1] * V_per[1] + V_per[2] * V_per[2]));
            for (unsigned int k = 0; k < 3; k++) {
                V_per[k] = V_per_abs * V_per[k] / v_per_aux_abs;
            }
			double V[3] = { V_par[0] + V_per[0], V_par[1] + V_per[1], V_par[2] + V_per[2] };

			// paso las velocidades
			vr[IC_gridsize*j+i] = V[0]; vtheta[IC_gridsize*j+i]=V[1]; vz[IC_gridsize*j+i]=V[2];
		}
	}
	
}

void Init_rv(double *rr,double *rq, double *rz, double *vr,double *vq,double *vz,double *tiempo,int npart){ 
  double  rr1,qqq,zz1,vr1,vq1,vz1,time,pitch,aux;
  int ind,i,aux_i;
  FILE *archi;

  archi=fopen("init_uni22.txt","r");


  for(i=0;i<npart;i++)
  //	 while(!feof(archi))
	{
	  fscanf(archi,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n",&ind,&time,&rr1,&qqq,&zz1,&vr1,&vq1,&vz1,&aux,&aux,&pitch,&aux_i);

	  rr[i]=rr1;
	  rq[i]=qqq;
	  rz[i]=zz1;

	  // xx[i]=rr1*cos(qqq);
	  //yy[i]=rr1*sin(qqq);
	  //zz[i]=zz1;
	  //vx=vr*cos(q)-vq*sin(q)
	  //vy=vr*sin(q)+vq*cos(q)

	  //vx[i]=vr1*cos(qqq)-vq1*sin(qqq);
	  //vy[i]=vr1*sin(qqq)+vq1*cos(qqq);
	  //vz[i]=vz1;
	   vr[i]=vr1;
	   vq[i]=vq1;
	   vz[i]=vz1;
           *tiempo=time;  
	  //Posicion puntual:
	  //r[0] = hR0 + (*x);
	  //r[1] = 0.0;
	  //r[2] = 0.0;
	}

  fclose(archi);
}

 
 /* ****** Coordenadas cilindricas ************ */
/*
void Init_v( double *v,double *R){	
  double B[3],E[3],R1, R2, theta, phi,bmod,pitch,rr,qq,zz,br,bq,bz,time,y;
  double vpar,vper,vper1,vper2,vmod,vr,vq,vz,vper_r,vper_q,vper_z,modper,ranq,ranz,ranr,modran,s_flux;

	double PI = 3.1415926535897932385;
	R1 = ran2(&init);
	R2 = ran2(&init);

	theta=acos(2.0*(R1-0.50));
	phi=2.0*PI*R2;

	//theta=pi/4.0d0;
	//phi=pi/4.0d0;

	// Esto esta en cartesianas, pero como es isotrópico
	// no hay problemas en usarlo con coord cilindricas.

	rr=R[0];
	qq=R[1];
	zz=R[2];

	//	v[0]=sin(theta)*cos(phi);
	//	v[1]=sin(theta)*sin(phi);
	//v[2]=cos(theta);

	pitch=1.0;
	B_Asdex(rr,zz, &B[0], &s_flux);
	//B_Asdex(&B[0],&E[0],rr,qq,zz,time,y);
	//	magnetic_equil(&B[0],rr,qq,zz);
	br=B[0];
	bq=B[1];
	bz=B[2];
	bmod=sqrt(br*br+bq*bq+bz*bz);
	
	// calculo la direccion  vper
	ranr=ran2(&init)-0.5;
	ranq=ran2(&init)-0.5;
	ranz=ran2(&init)-0.5;
          
	modran=sqrt(ranr*ranr+ranq*ranq+ranz*ranz);
	ranr=ranr/modran;
	ranq=ranq/modran;
	ranz=ranz/modran;
	vper=sqrt(1-pitch*pitch); 

	vper_r=(ranq*bz-ranz*bq)/bmod;
	vper_q=(br*ranz-bz*ranr)/bmod;
	vper_z=(ranr*bq-br*ranq)/bmod;
            
	modper=sqrt(vper_r*vper_r+vper_q*vper_q+vper_z*vper_z);
            
	    //	    if(vper==0)
	    // modper=1;
	    // a pitch=1 le sumo una componente gaussiana
	    // a vper le sumo una gaussiana
	    //pitch=pitch+v[0];
	    // vper=v[1];

	//	vper=dsqrt(1-pitch*pitch);
	
	if(vper==0)
	  modper=1.0;
            
	v[0]=pitch*br/bmod+vper*vper_r/modper;
	v[1]=pitch*bq/bmod+vper*vper_q/modper;
	v[2]=pitch*bz/bmod+vper*vper_z/modper;

	/* v[0]=0.0; */
	/* v[1]=0.0; */
	/* v[2]=1.0; * /
}
*/

/*
void Init_v_old( double *v,double *R){
	double R1, R2, theta, phi;
	double PI = 3.1415926535897932385;
	double v_mod;

	R1 = ran2(&init);
	R2 = ran2(&init);

	theta=acos(2.0*(R1-0.50)); //isotropia

	phi=2.0*PI*R2;


	v[0]=sin(theta)*cos(phi);
	v[1]=sin(theta)*sin(phi);
	v[2]=cos(theta);
}
*/



/*
void Init_v_old2( double *v,double *R){
  double R1, R2, theta, phi,B[3],bmod,br,bq,bz,pitch,ranr,ranq,ranz,modran,vper,vper_r,vper_q,vper_z,modper,vpar;

	double PI = 3.1415926535897932385;


	campo_mag(&B[0],R[0],R[1],R[2]);
	br=B[0];
	bq=B[1];
	bz=B[2];
	bmod=sqrt(br*br+bq*bq+bz*bz);
	
	pitch=1.0;

	/* vmod= sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);  * / 
	// calculo la direccion  vper
	ranr=ran2(&init)-0.5;
	ranq=ran2(&init)-0.5;
	ranz=ran2(&init)-0.5;
          
	modran=sqrt(ranr*ranr+ranq*ranq+ranz*ranz);
	ranr=ranr/modran;
	ranq=ranq/modran;
	ranz=ranz/modran;
	    
	vper=sqrt(1-pitch*pitch); 

	vper_r=(ranq*bz-ranz*bq)/bmod;
	vper_q=(br*ranz-bz*ranr)/bmod;
	vper_z=(ranr*bq-br*ranq)/bmod;
            
	modper=sqrt(vper_r*vper_r+vper_q*vper_q+vper_z*vper_z);
	
	if(vper==0)
	  modper=1;

	v[0]=pitch*br/bmod+vper*vper_r/modper;
	v[1]=pitch*bq/bmod+vper*vper_q/modper;
	v[2]=pitch*bz/bmod+vper*vper_z/modper;

//	vpar = (vr*B[0]+vq*B[1]+vz*B[2])/bmod; 
	  
	/* vmod2= (vr*vr+vq*vq+vz*vz);  */
	/* vper=sqrt(1-vpar*vpar/(vmod2)); */
	/* vmod2=sqrt(vmod2); */

	/* v[0]=vr; */
	/* v[1]=vq; */
	/* v[2]=vz; * /
}
*/

/*
void Init_r(double *r){
  double phi,xx,zz,rr;
  
	double PI = 3.1415926535897932385;

	
	phi = 2.0*PI*ran2(&init);
	rr=1.0;
	//	*x = 0.20;
	while(rr>0.6){
	xx= 2.0*(ran2(&init)-0.5);
	zz= 2.0*(ran2(&init)-0.5);
	// Como hay simetria azimutal, considero y=0.
	rr= sqrt(xx*xx+zz*zz);
	}
	r[0] = hR0 +xx;
	r[1] = phi; 				//Symetry
	r[2] = zz;
	
	/* //Posicion puntual: */
	/* r[0] = hR0 + (*x); */
	/* r[1] = 0.0; */
	/* r[2] = 0.0; * /
}
*/

/*
void Init_gc(struct Part *He, double y){
	//this function initializes the particle position
	// using a given guiding center and the particle velocity.
	int i;
	double vp[3];		// velocidad perpendicular.
	double e[3];			// vector unitario.
	double B[3],E[3];
	
	double rg[3];
	rg[0] = 3.8;
	rg[1] = 0.0;
	rg[2] = 0.0;
	
	//to avoid nan results
	(*He).r[0] = rg[0];	(*He).r[1] = rg[1]; 	(*He).r[2]=rg[2];
	//-----------------------

	if((*He).q == 0) return;

	magnetic_field(&B[0],&E[0], rg[0],rg[1],rg[2],(*He).time,y);

	double Bmod = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);

	for(i=0;i<3;i++)
		vp[i] = (*He).v[i]*B[i]/Bmod;			// Vel paralela.
	for(i=0;i<3;i++)
		vp[i] = (*He).v[i] - vp[i];			// Vel perpendicular
	
	double vpmod = sqrt(vp[0]*vp[0] + vp[1]*vp[1] + vp[2]*vp[2]);
	double rho = y*((double)(*He).Z)/((double)(*He).q)*vpmod/Bmod;
	
	e[0] = -((*He).v[1]*B[2] - (*He).v[2]*B[1]);
	e[1] = -((*He).v[2]*B[0] - (*He).v[0]*B[2]);
	e[2] = -((*He).v[0]*B[1] - (*He).v[1]*B[0]);
	
	if (rho < 1.0E-8)
		return; //devuelve, como posicion, el centro de giro.

	for(i=0;i<3;i++)
		e[i] = e[i]/(vpmod*Bmod);

	// Posición de la particula:
	// En coordenadas cilindricas tengo que dar (r, theta, z)
	
	for(i=0;i<3;i++)
		(*He).r[i] = rg[i] + rho*e[i];

}
*/

//da la proyección (v_paralela) y actualiza el valor del flujo. Es un poco confuso. (previamente la función estaba en el archivo principal)
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

//__device__ void RK46_NL(struct Part *He,double y,double *B1r1,double *B1r2,double *B1z1,double *B1z2){

__device__ void RK46_NL(struct Part *He,double y){
	  // y = gamma = y
	  //printf("Adentro de 'RK46_NL', v=%f\t", (*He).v[0]); 

  double u[6]; 	// velocidades y posiciones
  double w[6];	// vel./pos. intermedias
  double tw;	// tiempo intermedio;
  double q_Z;
  int i,j;
		//factor gamma:
	//double y = 14.457*sqrt(mu*Ep_MeV)/( Zp*B_0*a_cm);
	//double y = 0.02552;
  q_Z = (double)(*He).q/Zp;
  // q_Z = (double)(*He).q;
	//Vector a la funcion Right-hand side
  double F[6];
		// Constantes del método integrador
  double a[6],b[6],c[6];
  a[0] = 0.0;					b[0]=0.032918605146;	c[0]=0.0;
  a[1] = -0.737101392796;		b[1]=0.823256998200;	c[1]=0.032918605146;
  a[2] = -1.634740794341;		b[2]=0.381530948900;	c[2]=0.249351723343;
  a[3] = -0.744739003780;		b[3]=0.200092213184;	c[3]=0.466911705055;
  a[4] = -1.469897351522;		b[4]=1.718581042715;	c[4]=0.582030414044;
  a[5] = -2.813971388035;		b[5]=0.27;				c[5]=0.847252983783;
	
	// Condiciones iniciales:
  u[0] = (*He).v[0]; // v_{\rho}
  u[1] = (*He).v[1]; // v_{\theta}
  u[2] = (*He).v[2]; // v_{z}
  u[3] = (*He).r[0]; // r_{\rho}
  //printf("RK iteraciones r[0]: %f", u[3]);
  u[4] = (*He).r[1]; // r_{\theta}
  u[5] = (*He).r[2]; // r_{z}
  for(i=0;i<6;i++)
    w[i] = 0.0;

  for(i=0;i<6;i++){	//etapas
	  //	tw = ((double)n + c[i])*Dt;
    tw= (*He).time+c[i]*Dt;  
    //    RHS_cil(&F[0],&u[0],y,q_Z,tw,B1r1,B1r2,B1z1,B1z2);
     RHS_cil(&F[0],&u[0],y,q_Z,tw);
    for(j=0;j<6;j++){	// variables (pos./vel.)
      w[j] = a[i]*w[j] + Dt*F[j];
      u[j] = u[j] + b[i]*w[j];
    }
  }
	
  // Actualizo Vel./pos.:
  for(j=0;j<3;j++){
    (*He).v[j] = u[j];
    (*He).r[j] = u[j+3];
  }
  (*He).E_keV = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])*dEp_MeV*1000.0;
    (*He).time = (*He).time + Dt;
  //(*He).time =tw;

    #ifdef Z_1  // No lo entiendo bien, pero está en lo de Cesar
		if ((*He).q==0) {
		(*He).timeAt = (*He).timeAt + Dt*ta;
		}
		// ta tiempo de ciclotrón
		// Así como está luego de ionizarse timeAt no cambia luego de ionizar
	#endif

  return;

	//free(pF);
	//pF = NULL;
}
//__device__ void RHS_cil(double *FF, double *u, double y, double q_Z, double time,double *B1r1,double *B1r2,double *B1z1,double *B1z2){
	// Right hand side del RK
__device__ void RHS_cil(double *FF, double *u, double y, double q_Z, double time){
	// el tiempo esta agregado para incluir perturbaciones.
  double B[3], E[3],Bzper,Brper,flux; // Campo magnetico y E
	//		double rc[3];
	//		rc[0]=u[3];
	//		rc[1]=u[4];
	//		rc[2]=u[5];


	//	B_circular(&B[0], u[3],u[4],u[5]);

    //aca habria que pasar el gamma=y
	// actualiza el campo para la nueva posición, no se por qué está definido así pero bueno
    magnetic_field(&B[0],&E[0], u[3],u[4],u[5]);  // Donde está definida?, parece que Mfield.h, cómo funciona eso? (PREGUNTAR)
    //magnetic_perturbation( u[3],u[4],u[5],&B1r1[0],&B1r2[0],&B1z1[0],&B1z2[0],&Brper,&Bzper,time); 
  
    // B[0]=B[0]+Brper;
    // B[2]=B[2]+Bzper;

	// (PREGUNTAR) -> la q_Z no debería multiplicar al E? Capaz no porque no se la dinamica con E
    FF[0] = y*u[1]*u[1]/u[3] + q_Z*(u[1]*B[2]-u[2]*B[1])+E[0];
	FF[1] =  - y*u[0]*u[1]/u[3] + q_Z*(u[2]*B[0]-u[0]*B[2])+E[1];
	FF[2] = q_Z*(u[0]*B[1] - u[1]*B[0])+E[2];
	FF[3] = y*u[0];  //dr/dt
	FF[4] = y*u[1]/u[3];  //dtheta/dt
	FF[5] = y*u[2];  //dz/dt
	return;
}

__host__ void H_RK46_NL(struct Part *He, double y){
  //printf("Dentro del RK: %f \t %f \t %f \n",(*He).r[0], (*He).r[1], (*He).r[2]);
	  // y = gamma = y
  double u[6]; 	// velocidades y posiciones
  double w[6];	// vel./pos. intermedias
  double tw;	// tiempo intermedio;
  double q_Z;
  int i,j;
		//factor gamma:
	//double y = 14.457*sqrt(mu*Ep_MeV)/( Zp*B_0*a_cm);
	//double y = 0.02552;
  q_Z = (double)(*He).q/hZp;
  // q_Z = (double)(*He).q;
	//Vector a la funcion Right-hand side
  double F[6];
		// Constantes del método integrador
  double a[6],b[6],c[6];
  a[0] = 0.0;					b[0]=0.032918605146;	c[0]=0.0;
  a[1] = -0.737101392796;		b[1]=0.823256998200;	c[1]=0.032918605146;
  a[2] = -1.634740794341;		b[2]=0.381530948900;	c[2]=0.249351723343;
  a[3] = -0.744739003780;		b[3]=0.200092213184;	c[3]=0.466911705055;
  a[4] = -1.469897351522;		b[4]=1.718581042715;	c[4]=0.582030414044;
  a[5] = -2.813971388035;		b[5]=0.27;				c[5]=0.847252983783;
	
	// Condiciones iniciales:
  u[0] = (*He).v[0]; // v_{\rho}
  u[1] = (*He).v[1]; // v_{\theta}
  u[2] = (*He).v[2]; // v_{z}
  u[3] = (*He).r[0]; // r_{\rho}
  //printf("RK iteraciones r[0]: %f", u[3]);
  u[4] = (*He).r[1]; // r_{\theta}
  u[5] = (*He).r[2]; // r_{z}
  for(i=0;i<6;i++)
    w[i] = 0.0;

  for(i=0;i<6;i++){	//etapas
	  //	tw = ((double)n + c[i])*Dt;
    tw= (*He).time+c[i]*hDt;  
    //    RHS_cil(&F[0],&u[0],y,q_Z,tw,B1r1,B1r2,B1z1,B1z2);
     H_RHS_cil(&F[0],&u[0],y,q_Z,tw);
    for(j=0;j<6;j++){	// variables (pos./vel.)
      w[j] = a[i]*w[j] + hDt*F[j];
      u[j] = u[j] + b[i]*w[j];
    }
  }
	
  // Actualizo Vel./pos.:
  for(j=0;j<3;j++){
    (*He).v[j] = u[j];
    (*He).r[j] = u[j+3];
  }
  (*He).E_keV = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])*Ep_MeV*1000.0;
    (*He).time = (*He).time + hDt;
  //(*He).time =tw;

    

  return;
	//free(pF);
	//pF = NULL;
}

//__device__ void RHS_cil(double *FF, double *u, double y, double q_Z, double time,double *B1r1,double *B1r2,double *B1z1,double *B1z2){
	// Right hand side del RK
__host__ void H_RHS_cil(double *FF, double *u, double y, double q_Z, double time){
	// el tiempo esta agregado para incluir perturbaciones.
  double B[3], E[3],Bzper,Brper,flux; // Campo magnetico y E
	//		double rc[3];
	//		rc[0]=u[3];
	//		rc[1]=u[4];
	//		rc[2]=u[5];
	//	B_circular(&B[0], u[3],u[4],u[5]);

    //aca habria que pasar el gamma=y
	// actualiza el campo para la nueva posición, no se por qué está definido así pero bueno
    magnetic_field(&B[0],&E[0], u[3],u[4],u[5]);  // Donde está definida?, parece que Mfield.h, cómo funciona eso? (PREGUNTAR)
    //magnetic_perturbation( u[3],u[4],u[5],&B1r1[0],&B1r2[0],&B1z1[0],&B1z2[0],&Brper,&Bzper,time); 
  
    // B[0]=B[0]+Brper;
    // B[2]=B[2]+Bzper;

	// (PREGUNTAR) -> la q_Z no debería multiplicar al E? Capaz no porque no se la dinamica con E
    FF[0] = y*u[1]*u[1]/u[3] + q_Z*(u[1]*B[2]-u[2]*B[1])+E[0];
	FF[1] =  - y*u[0]*u[1]/u[3] + q_Z*(u[2]*B[0]-u[0]*B[2])+E[1];
	FF[2] = q_Z*(u[0]*B[1] - u[1]*B[0])+E[2];
	FF[3] = y*u[0];  //dr/dt
	FF[4] = y*u[1]/u[3];  //dtheta/dt
	FF[5] = y*u[2];  //dz/dt
	return;
}





/*
__device__ void Boris_c(struct Part *He,double y){
	
	//double Bcil[3];
	double Bc[3];
	double t[3];
	double E[3];
	double vp[3];
	double t2;
	double s;
	double v_dot_t;
	double c_ov_v0 = 21.645*sqrt(mu/dEp_MeV);
	//double y = 54.09/( (double)(*He).q*B_0*a_cm);
	
	//double vc[3];
	//double rcil[3];
	double rc[3];
	
	rc[0] = (*He).r[0] + y*(0.5*Dt)*(*He).v[0];
	rc[1] = (*He).r[1] + y*(0.5*Dt)*(*He).v[1]; 
	rc[2] = (*He).r[2] + y*(0.5*Dt)*(*He).v[2];
	
	// Los campos se calculan en t + Dt/2!
	//magnetic_field(&Bc[0], rc[0],rc[1],rc[2]);
	magnetic_field(&Bc[0],&E[0], rc[0],rc[1],rc[2],(*He).time,y);
	// Campo electrico nulo.
	E[0] = 0.0; E[1] = 0.0 ; E[2] = 0.0;
	
		
	/* ********** Velocidad  ***********
	 *********************************** /
	// calculo v(i+1/2):

	//hugo falta un 0.5 para el Dt del E
	vp[0] = (*He).v[0] + ((double)(*He).q/Zp)*c_ov_v0*(E_0/B_0)*E[0]*Dt
					   + ((double)(*He).q/Zp)*(0.5*Dt)*((*He).v[1]*Bc[2] - (*He).v[2]*Bc[1]);
	//hugo, aca falta un Dt?
	vp[1] = (*He).v[1] + ((double)(*He).q/Zp)*c_ov_v0*(E_0/B_0)*E[1]
					   + ((double)(*He).q/Zp)*(0.5*Dt)*((*He).v[2]*Bc[0] - (*He).v[0]*Bc[2]);

	//hugo, aca falta un Dt?

	vp[2] = (*He).v[2] + ((double)(*He).q/Zp)*c_ov_v0*(E_0/B_0)*E[2]
					   + ((double)(*He).q/Zp)*(0.5*Dt)*((*He).v[0]*Bc[1] - (*He).v[1]*Bc[0]);
		
	t[0] = ((double)(*He).q/Zp)*0.5*Dt*Bc[0];
	t[1] = ((double)(*He).q/Zp)*0.5*Dt*Bc[1];
	t[2] = ((double)(*He).q/Zp)*0.5*Dt*Bc[2];

	t2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
	s = 1.0/(1.0 + t2);
	v_dot_t = vp[0]*t[0] + vp[1]*t[1] + vp[2]*t[2];

	// calculo v(t+Dt):
	(*He).v[0] = s*(vp[0] + v_dot_t*t[0] + (vp[1]*t[2] - vp[2]*t[1]) );
	(*He).v[1] = s*(vp[1] + v_dot_t*t[1] + (vp[2]*t[0] - vp[0]*t[2]) );
	(*He).v[2] = s*(vp[2] + v_dot_t*t[2] + (vp[0]*t[1] - vp[1]*t[0]) );
	
	// Posiciones en t + Dt
	(*He).r[0] = rc[0] + y*(0.5*Dt)*(*He).v[0];
	(*He).r[1] = rc[1] + y*(0.5*Dt)*(*He).v[1]; 
	(*He).r[2] = rc[2] + y*(0.5*Dt)*(*He).v[2];
	
	(*He).E_keV = ( (*He).v[0]*(*He).v[0] +
			(*He).v[1]*(*He).v[1] +
        		(*He).v[2]*(*He).v[2] )*dEp_MeV*1000.0;
	(*He).time = (*He).time + Dt; //time in msec
}
*/


__host__ __device__ void centro_giro(struct Part *He, double *rg, double y){
	int i;
	double vp[3];		// velocidad perpendicular.
	double e[3];			// vector unitario.
	double B[3],E[3];
	
	rg[0] = (*He).r[0];
	rg[1] = (*He).r[1];
	rg[2] = (*He).r[2];
	
	if((*He).q == 0) return;  // partícula neutra

	magnetic_field(&B[0],&E[0], (*He).r[0],(*He).r[1],(*He).r[2]);

	double Bmod = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);

	for(i=0;i<3;i++)
		vp[i] = ((*He).v[i])*B[i]/Bmod;			// Vel paralela.
	for(i=0;i<3;i++)
		vp[i] = (*He).v[i] - vp[i];			// Vel perpendicular
	
	double vpmod = sqrt(vp[0]*vp[0] + vp[1]*vp[1] + vp[2]*vp[2]);
	double rho = y*((double)(*He).Z)/((double)(*He).q)*vpmod/Bmod;
	
	e[0] = (*He).v[1]*B[2] - (*He).v[2]*B[1];
	e[1] = (*He).v[2]*B[0] - (*He).v[0]*B[2];
	e[2] = (*He).v[0]*B[1] - (*He).v[1]*B[0];
	
	if (rho < 1.0E-8)
		return; //devuelve la posicion de la particula.

	for(i=0;i<3;i++)
		e[i] = e[i]/(vpmod*Bmod);

	// Posición del centro de giro:
	for(i=0;i<3;i++)
		rg[i] = (*He).r[i] + rho*e[i];
}


	
/* ***** Density profiles **************/

__device__ double n_ei(double x){
  //parabola invertida en unidades de 10**20m-3
  // return -0.2*x*x+0.8;
  return (-0.2*x*x+0.8)*0.5*(1-tanh(50*(x-0.8)));
	/*// Funcion tipo tanh .
	double x0 = 0.95; 	// Posición barrera.
	double Dx = 0.02;	// Mide el Ancho de la barrera
	
	return 0.5*(1.0 + tanh((-x+x0)/Dx));
	*/
  //	return 1.0;
}


__device__ double f_ei(double x){
	/*// Funcion tipo tanh .
	double x0 = 0.95; 	// Posición barrera.
	double Dx = 0.02;	// Mide el Ancho de la barrera
	
	return 0.5*(1.0 + tanh((-x+x0)/Dx));
	*/
	return 1.0;
}
__device__ double Te(double x){
  // return 2.1*exp(-x*x*1.65289256198347);
  return 2.1*exp(-x*x*1.65289256198347)*0.5*(1-tanh(50*(x-0.8)));

	/*// Son tres tramos de rectas:
	// Intenta simular un modo-H
	// -> T1[0,a] 		a -> barrera de trasporte
	// -> T2[a,b]		b -> separatriz
	// -> T3[b,c]		c -> pared
	double a = 0.9;
	double b = 1.0;
	double c = 1.1;
	
	double T0 = 10.0; 	// keV
	double Ta = 5.0;  	// keV
	double Tb = 0.1; 	// keV
	double Tc = 0.01;	// keV
	
	if(x < a){
		return ((Ta - T0)/a)*x + T0;
	}else if(x>=a && x<b){
		return ((Tb - Ta)/(b-a))*(x-a) + Ta;
	}else{
		return ((Tc - Tb)/(c-b))*(x-b) + Tb;
	}*/
 //	return 5.0;
}

__device__ double f_n(double x){
	// Funcion tipo tanh .
	//double x0 = 1.0; 	// Posición barrera.
	//double Dx = 0.02;	// Mide el Ancho de la barrera
	//
	//return 0.5*(1.0 + tanh((x-x0)/Dx));
	return 1.0; //densidades constantes.
}

//============================================================
// Gaussian Random distribution
// See Num. Recipies in C. p289.
__host__ __device__ void Ran_gauss(double *RG, double sigma){
    double pi = 3.1415926535897932385;
    double y1, y2;
    y1 = sigma*sqrt(-2.0*log(RG[0]))*cos(2.0*pi*RG[1]);
    y2 = sigma*sqrt(-2.0*log(RG[0]))*sin(2.0*pi*RG[1]);

    RG[0] = y1;
    RG[1] = y2;
}



// ==== funcion ran2 ============
 
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran2(long *idum)
{
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[NTAB];
	float temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		idum2=(*idum);
		for (j=NTAB+7;j>=0;j--) {
			k=(*idum)/IQ1;
			*idum=IA1*(*idum-k*IQ1)-k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy=iv[0];
	}
	k=(*idum)/IQ1;
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if (*idum < 0) *idum += IM1;
	k=idum2/IQ2;
	idum2=IA2*(idum2-k*IQ2)-k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) return (double)RNMX;
	else return (double)temp;
}
#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
/* (C) Copr. 1986-92 Numerical Recipes Software p,{2. */

void CrossProduct(double v_A[3], double v_B[3], double c_P[3]) {
    c_P[0] = v_A[1] * v_B[2] - v_A[2] * v_B[1];
    c_P[1] = -v_A[0] * v_B[2] + v_A[2] * v_B[0];
    c_P[2] = v_A[0] * v_B[1] - v_A[1] * v_B[0];
}
