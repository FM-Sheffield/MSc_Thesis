/* ******* Magnetic Field *************/
//Magnetic field selector function:
__host__ __device__ void magnetic_field(double *B,double *E, double r, double qq, double z);
//__host__ __device__ void magnetic_perturbation(double rr,double rq,double rz,double *B1r1,double *B1r2,double *B1z1,double *B1z2,double *Brper,double *Bzper,double time);
//magnetic_pertubation( u[3],u[4],u[5],&B1r1[0],&B1r2[0],&B1z1[0],&Brper,&Bzper);

//__host__ __device__ void B_Asdex(double *B, double *E, double r, double qq, double z,double time,double y);
//__host__ __device__ void B_Asdex(double r,double z,double *B_eq,double *s_flux);
 /******* Cylindrical coordinates *******/
// Circular cross section:
//__host__ __device__ void B_circular(double *B, double *E, double r, double qq, double z,double time,double y);

//__host__ __device__ void B_constante(double *B, double x, double y, double z);

__host__ __device__ void B_Analitico(double R,double Z,double *B,double *s_flux);
__host__ __device__ double Psi_analitico(double R, double Z);

//__device__ double Psi(double x, double y, double z);




// Setea los campos en r, theta (qq), z
//__host__ __device__ void magnetic_field(double *B, double *E, double r, double qq, double z,double time,double y)
__host__ __device__ void magnetic_field(double *B, double *E, double r, double qq, double z){
  //	B_constante(B, x, y, z);
//    B_circular(B, E,r, qq, z,time,y);
    double s_flux=0;
   //B_Asdex(B, E,r, qq, z,time,y);	
    B_Analitico(r,z,B,&s_flux);
    E[0]=0.0;
    E[1]=0.0;
    E[2]=0.0;
}


/*
__host__ __device__ void magnetic_perturbation(double rr,double rq,double rz,double *B1r1,double *B1r2,double *B1z1,double *B1z2,double *Brper,double *Bzper,double time)
{
  double rri,zzi,p=0,q=0,fff,phii,Bzper1,Brper1;
  int ii,jj;
  double omega=-0.e-4;
  //copiado de ~/Documentos/mauricio2/hugo2/Codigos/TrayectoriasAsdex/B1_22$
  //ojo que si estos numeros estan mal no puedo leer bien la matriz
  int N_r=401;
  int N_z=401;
  
  double z0 = -0.9;
  double r0= 1.1;
  double z1 = 0.9;
  double r1= 2.2;
  double dr=(r1-r0)/(N_r-1.);
  double dz=(z1-z0)/(N_z-1.);
  
  phii=rq+omega*time;


  //paso a la grilla sin adimensionalizar
  rri=0.5*rr;
  zzi=rz*0.5;
  
  if(rri<r0 || rri>r1 || zzi <z0 || zzi>z1)
    {
      ii=0;
      jj=0;
      q=0;
      p=0;
      fff=0.0;
    }
  else
    {
      jj=(rri-r0)/dr;
      ii=(zzi-z0)/dz;
      p=(rri-r0-jj*dr)/dr; 
      q=(zzi-z0-ii*dz)/dz;
      fff=0.0; //este seria mejor pasarlo afuera, a rhs_cil
    }
  /* B_Asdex(ri,zi, B_equilibrio, &s_flux);		  */
  
  /* Br=B_equilibrio[0]; */
  /* Bt=B_equilibrio[1];				 */
  /* Bz=B_equilibrio[2]; *
  
  Brper1=    (1.0-p)*(1.0-q)*( B1r1[ii*N_r + jj]*cos(phii)+B1r2[ii*N_r + jj]*sin(-phii) )  + q*(1.0-p)*( B1r1[(ii+1)*N_r+ jj]*cos(phii)+B1r2[(ii+1)*N_r+jj]*sin(-phii) ) +
    p*(1.0-q)*( B1r1[ii*N_r + jj+1]*cos(phii)+B1r2[ii*N_r + jj+1]*sin(-phii) ) + p*q*( B1r1[(ii+1)*N_r + jj+1]*cos(phii)+B1r2[(ii+1)*N_r + jj+1]*sin(-phii) ) ;
  
  Bzper1= (1.0-p)*(1.0-q)*( B1z1[ii*N_r + jj]*cos(phii)+B1z2[ii*N_r + jj]*sin(-phii) )  + q*(1.0-p)*( B1z1[(ii+1)*N_r+ jj]*cos(phii)+B1z2[(ii+1)*N_r+ jj]*sin(-phii) ) +
    p*(1.0-q)*( B1z1[ii*N_r + jj+1]*cos(phii)+B1z2[ii*N_r + jj+1]*sin(-phii) ) + p*q*( B1z1[(ii+1)*N_r + jj+1]*cos(phii)+B1z2[(ii+1)*N_r + jj+1]*sin(-phii) ) ;

  

  *Brper=Brper1*fff;
  *Bzper=Bzper1*fff;
  
}
*/

// Global Parameters (analytic equilibrium coeffs)
const double M = 0.01;
// Negative T_2 - DIIID - test cand 4 (A=0.567 , V=3.71):
//const double C1 = -0.004892540131; const double C2 = 0.2072398556; const double C3 = 0.5556866341; const double C4 = 0.1702685657; 
//const double C5 = 0.2663315198; const double C6 = 0.01729018624; const double C7 = -0.05564634276; 
//const double C8 = -0.03619877453; const double S1 = -0.5395195344810759; const double S2 = -0.004129137273799056;

//Positive T_2 - DIIID - test cand 4 (A = 0.568, V = 3.405):
//const double C1 = 0.001812141542; const double C2 = 0.4717347435; const double C3 = 0.5313153095; const double C4 = 0.4407871704; 
//const double C5 = 0.4130720551; const double C6 = 0.02982725318; const double C7 = -0.04191041218; const double C8 = 0.0214024026;
//const double S1 = -0.6244922921; const double S2 = -0.02969385881;

//Negative T - DIIID - cand 5 (B=2T, A = 0.5675, V = 3.7152):
//const double S1 = -0.4995410553; const double S2 = 0.09395498849;    
//const double C1 = -0.01154271581; const double C2 = 0.2125359453; const double C3 = 0.5489561266; const double C4 = 0.1482830648; 
//const double C5 = 0.2636284353; const double C6 = 0.01753591288; const double C7 = -0.05560087368; const double C8 = -0.03804031965;

//Positive T - DIIID - cand 5 (B=2T, A = 0.5676, V = 3.4043):
const double S1 = -0.5831152318; const double S2 = 0.05676267719;    
const double C1 = -0.002190157406; const double C2 = 0.5580341791; const double C3 = 0.6618636006; const double C4 = 0.4918686431; 
const double C5 = 0.5099540459; const double C6 = 0.02875339697; const double C7 = -0.05279197327; const double C8 = 0.02561018266;


// Calcula el flujo poloidal analitico en r, z
__host__ __device__ double Psi_analitico(double R, double Z) {
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
__host__ __device__ void B_Analitico(double R,double Z,double *B,double *s_flux){
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
	*s_flux= Psi_analitico(R, Z);
}


/*
__host__ __device__ void B_Asdex(double r,double z,double *B,double *s_flux){

	r=r*0.5;
	z=z*0.5;
  double T_a=15.2329; 
  double p_a=sqrt(T_a); 
  double q_a=p_a/2.0;
  double nu_a=p_a*sqrt(3.0/4.0);
  double BT0asdex=17.815757116271065;


	double cc1=0.4733, cc2=-0.2164,cc3=0.0, cc4=0.0, cc5=0.0, cc6=0.0,cc7=-0.06830, cc8=0.01220, cc9=0.1687;
	double cc10=0.8635, cc11=-1.0682, cc12=0.02166,cc13=-0.002662, cc14=0.1178, cc15=1.4008, cc16=-0.2656,cc17=1.3770, cc18=0.2468;


	double csp=cos(p_a*z);
	double snp=sin(p_a*z);
	double csq=cos(q_a*z);
	double snq=sin(q_a*z);
	double csnu=cos(nu_a*z);
	double snnu=sin(nu_a*z);
	double jb1p=j1(p_a*r);
	double jb1q=j1(q_a*r);	
	double jb1nu=j1(nu_a*r);
	double yb1q=y1(q_a*r);	
	double yb1nu=y1(nu_a*r);
	double rho=sqrt(r*r+z*z);
	double Br=0.0;
	Br=(-(r*jb1p*cc4-cc5*p_a*snp+cc6*p_a*csp+r*r*p_a*( -cc7*snp+cc8*csp ) - cc9*p_a*sin(p_a*rho)*(z/rho) + cc10*p_a*cos(p_a*rho)*(z/rho)+ r*jb1nu*(-q_a*cc11*snq+cc12*q_a*csq) +r*jb1q*( -cc13*nu_a*snnu +cc14*nu_a*csnu ) + r*yb1nu*( -cc15*q_a*snq+cc16*q_a*csq) + r*yb1q*(-nu_a*cc17*snnu + cc18*nu_a*csnu))*(1.0/r))/(BT0asdex);

	double jb0p=j0(p_a*r);
	double jb0q=j0(q_a*r);	
	double jb0nu=j0(nu_a*r);
	double yb0q=y0(q_a*r);	
	double yb0nu=y0(nu_a*r);		
	double Bz=0.0;
	Bz=(( 2.0*cc2*r  + jb1p*( cc3 + z*cc4) +r*(cc3+cc4*z)*( p_a*jb0p-(jb1p/r) ) +2.0*r*( cc7*csp+cc8*snp) - cc9*sin(p_a*rho)*((p_a*r)/rho) + cc10*cos(p_a*rho)*((p_a*r)/rho) + jb1nu*(cc11*csq+cc12*snq) +r*(cc11*csq +cc12*snq)*( nu_a*jb0nu-(jb1nu/r) ) + jb1q*(cc13*csnu + cc14*snnu) + r*(cc13*csnu + cc14*snnu)*( q_a*jb0q-(jb1q/r) ) + yb1nu*(cc15*csq+cc16*snq) +r*(cc15*csq+cc16*snq)*( nu_a*yb0nu-(yb1nu/r) ) + yb1q*( cc17*csnu +cc18*snnu) + r*( cc17*csnu +cc18*snnu)*( q_a*yb0q-(yb1q/r) )   )*(1.0/r) )/(BT0asdex) ;
	
	double Bt=0.0;	
	
	double u_a=-(cc1*T_a);
	double F0_a= 30.4;	
	double Psi= cc1 + cc2*r*r+ r*jb1p*(cc3+cc3*z) + cc5*csp + cc6*snp + r*r*(cc7*csp + cc8*snp) +cc9*cos(p_a*rho) + cc10*sin(p_a*rho) + r*jb1nu*(cc11*csq +cc12*snq) + r*jb1q*(cc13*csnu +cc14*snnu) + r*yb1nu*(cc15*csq + cc16*snq) + r*yb1q*(cc17*csnu+cc18*snnu)  ;
	Bt= ((sqrt(T_a*Psi*Psi+2.0*u_a*Psi+ F0_a*F0_a ))/r)/(BT0asdex) ;
	B[0]=Br;
	B[1]=Bt;
	B[2]=Bz;

	*s_flux= Psi;
}
*/




/* __host__ __device__ void B_circular(double *B, double xc, double yc, double zc,double time){ */
/* // Coincide con el de Ricardo. */
/* 	// Necesito coord. toroidales: */
	
/* 	double R0 = R_cm/a_cm;		//(la defino nuevamente porque la fn es host - device!) */

/* 	double ep = 1.0/R0; */
/* 	double r = sqrt(xc*xc +yc*yc); */
/* 	// Usamos x para la coord toroidal. */
/* 	double x = sqrt((r-R0)*(r-R0) + zc*zc); */
/* 	double cos_ph = (r-R0)/x; */
/* 	double sin_ph = zc/x; */
/* 	double Bcil[3]; */
/* 	//if((r-R0)==0.0) */
/* 	//	printf("\n cos: %f \t sin: %f \n", cos_ph, sin_ph); */
	
	
/* 	// constantes: */
/* 	double k = 2.4048255577;		// 1er cero de la J_0. */
/* 	double p1 = 0.05;			// Ref: R. Farengo, Plasma Phys. Control. Fusion 54 (2012) 025007. */
/* 	double I0 = 1.0; */
/* 	double af = 4.0*p1/(ep*ep); */
/* 	double Bw = 0.17;					// Campo poloidal en x = 1, theta = 0. Ref: Idem p1. */
/* 	double I1 = sqrt(0.25*ep*ep*k*k-p1);	// Corriente orden 1. */
/* 	// calculo la constante c: */
/* 	double aux = -0.5*(ep/(1.0+ep))*k*j1(k)*(1.0 + 0.5*ep*(1.0+2.0*af/(k*k))); */
/* 	double C = Bw/aux;		//corroborado con el codigo de Ricardo. */
	
	 
/* 	// flujo poloidal: */
/* 	double Psi = C*j0(k*x) + ep*C*0.5*cos_ph*(x*j0(k*x) + (af/k)*j1(k*x)*(1.0-x*x)); */
	
/* 	/\* // Derivadas del flujo poloidal: */
/* 	double dPsi_x = -C*( k + 0.5*ep*cos_ph*(k*x + af/(k*x)*(1.0-x*x)+2.0*af*x/k) )*bessj1(k*x); */
/* 	double dPsi_x_2 = C*0.5*ep*cos_ph*(af*(1.0-x*x)+1.0)*bessj0(k*x); */
	
/* 	dPsi_x = dPsi_x + dPsi_x_2; */
/* 	double dPsi_ph = -C*0.5*ep*sin_ph*( x*bessj0(k*x) + (af/k)*(1.0-x*x)*bessj1(k*x) ); *\/ */

/* 	double dPsi_x = -C*k*j1(k*x) + 0.5*C*ep*cos_ph*( j0(k*x) - k*x*j1(k*x)*(1.0+2.0*af/(k*k))  */
/* 					+ af*(1.0-x*x)*(j0(k*x)-1.0/(k*x)*j1(k*x)) ); */
/* 	double dPsi_ph = -0.5*C*ep*sin_ph*(x*j0(k*x) + (af/k)*j1(k*x)*(1.0-x*x)); */
	
/* 	//campo magnetico (cilindricas) */
/* 	Bcil[0] = -ep/(2.0*(1.0+ep*x*cos_ph))*(sin_ph*dPsi_x + (cos_ph/x)*dPsi_ph); */
/* 	  //pertubacion */
/* 	// */
       

/* 	Bcil[1] = 1.0/(1.0+ep*x*cos_ph)*sqrt(I0*I0 + I1*I1*Psi*Psi); */
/* 	Bcil[2] = ep/(2.0*(1.0+ep*x*cos_ph))*(cos_ph*dPsi_x - (sin_ph/x)*dPsi_ph); */

/* 	//campo magnetico (cartesianas) */
/* 	//	double theta = atan2(yc,xc); */
/* 	//	B[0] = Bcil[0]*cos(theta) - Bcil[1]*sin(theta); */
/* 	//B[1] = Bcil[0]*sin(theta) + Bcil[1]*cos(theta); */
/* 	//B[2] = Bcil[2]; */
/* 	B[0] = Bcil[0]; */
/* 	B[1] = Bcil[1]; */
/* 	B[2] = Bcil[2]; */
	


/* } */

    





/*
__host__ __device__ void B_constante(double *B, double xc, double yc, double zc){
	B[0] = 0.0;
	B[1] = 0.0;
	B[2] = 1.0;
}


__device__ double Psi(double xc, double y, double z){
// Coincide con el de Ricardo.
	// Necesito coord. toroidales:
	double R0 = R_cm/a_cm;
	double ep = 1.0/R0;
	double r = sqrt(xc*xc + y*y);
	// Aca usamos x para la cordenada toroidal
	double x = sqrt((r-R0)*(r-R0) + z*z);
	double cos_ph = (r-R0)/x;
		
	
	// constantes:
	double k = 2.4048255577;		// 1er cero de la J_0.
	double p1 = 0.05;			// Ref: R. Farengo, Plasma Phys. Control. Fusion 54 (2012) 025007.
	double af = 4.0*p1/(ep*ep);
	double Bw = 0.17;					// Campo poloidal en x = 1, theta = 0. Ref: Idem p1.
	// calculo la constante c:
	double aux = -0.5*(ep/(1.0+ep))*k*j1(k)*(1.0 + 0.5*ep*(1.0+2.0*af/(k*k)));
	double C = Bw/aux;		//corroborado con el codigo de Ricardo.
	
	 
	// flujo poloidal:
	double Psi = C*j0(k*x) + ep*C*0.5*cos_ph*(x*j0(k*x) + (af/k)*j1(k*x)*(1.0-x*x));
	return Psi;
}
*/

