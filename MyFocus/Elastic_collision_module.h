__device__ void Elastic_collisions(	struct Part *He,
					double Dt,
					int * i,
					long init,
					int tid);


__device__ void Elastic_collisions_PE_MC_euler (struct Part *He, double Dt, double * Ran_EC);
// (pitch - energy - Monte carlo algorithm)


__device__ void Elastic_collisions_PE_MC_KP2_ito (struct Part *He, double Dt, double * Ran_EC); //Klauder and Petersen

__device__ void Elastic_collisions_SV_euler (struct Part *He, double Dt, double * Ran_EC);

__device__ void coll_freq_par_perp(double *nu, double Eau, double Qa, double r, double th, double z, double * Ran_EC);
__device__ void coll_freq(double *nu, double Eau, double Qa, double r, double th, double z, double * Ran_EC);


/* **** Colisiones elásticas ******* */
__device__ void Elastic_collisions(struct Part *He, double Dt, int * i, long init, int tid){
	// Dt en realidad es Nec*Dt
	// printf("Adentro de 'Elastic_collisions', v=%f\t", (*He).v[0]);  

	//printf("Elastic Collision\n");
	//Random numbers initialization ------
	philox4x32_ctr_t   c={{}};
	philox4x32_key_t   k={{}};
	philox4x32_ctr_t   r;

	k.v[0] = tid; 		// thread id
	k.v[1] = init;		// global seed 
	c.v[0] = *i;		// global counter (iteration index)
	*i = *i + 1;
	//------------------------------------
	double Ran_EC[4];  // pitch-energy

	r = philox4x32(c, k);
	Ran_EC[0] = (double) u01_open_open_32_53(r.v[0]);
	Ran_EC[1] = (double) u01_open_open_32_53(r.v[1]);
	Ran_EC[2] = (double) u01_open_open_32_53(r.v[2]);
	Ran_EC[3] = (double) u01_open_open_32_53(r.v[3]);

	//ran_aux = Ran_EC[2];
	//Ran_gauss(&Ran_EC[2], 1.0);

	//Ran_EC[2] = ran_aux; //Solo para PE_MC_euler!!!!!

	//printf("%d \t %f \t %f \t %f \t %f \n", tid, Ran_EC[0], Ran_EC[1], Ran_EC[2], Ran_EC[3]);
	//Elastic_collisions_PE_MC_euler ( He, Dt, &Ran_EC[0]);
	//Elastic_collisions_PE_MC_KP2_ito ( He, Dt, &Ran_EC[0]);
	Elastic_collisions_SV_euler (He, Dt, &Ran_EC[0]);
	// printf("Right after the collision: v=%f\t", (*He).v[0]);  

}



__device__ void Elastic_collisions_PE_MC_euler (struct Part *He, double Dt, double * Ran_EC){
	/* pitch - energy - Monte Carlo implementation*/
	/* Utiliza la aproximacion de colisiones elasticas
		en la formulación centro de giro */
	//printf("Inside 'Elastic_collisions_PE_MC_euler': v=%f\t", (*He).v[0]);

	if((*He).q == 0) return;
	double r = (*He).r[0];
	double th = (*He).r[1];
	double z = (*He).r[2];
	double Qa = (double)(*He).q;

	double v_col[6]; 	//collision frequencies

	//conversions
	double Dt_au = Dt*ta*c_cgs/(137.0*a0_cgs);
	double v0_c;	// = v0/c
	double Ma = 1836.2*mu;						// Projectile mass
	v0_c = sqrt(2.0*(dEp_MeV/27.2E-6)/Ma)/137.0;

	double vmod = sqrt( (*He).v[0]*(*He).v[0] + (*He).v[1]*(*He).v[1] + (*He).v[2]*(*He).v[2] );
	double v_au = vmod*v0_c*137.0;

	double E_au = 0.5*Ma*v_au*v_au;

	double T_e = Te( r, th, z )/27.2E-3;
	double T_i = T_e;

	double B[3], E[3], Bmod;
	
	magnetic_field(&B[0], &E[0], (*He).r[0], (*He).r[1], (*He).r[2]);
	Bmod = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);


	//pitch inicial
	double cos_p = ((*He).v[0]*B[0] + (*He).v[1]*B[1] + (*He).v[2]*B[2])/(Bmod*vmod);
	double sin_p = sqrt(1.0 - cos_p*cos_p);

	// Parallel and Perpendicular vectors (respecto de B) ---------------------------
	double vpar[3];
	double vperp[3], vperp_mod;

	// Parallel
	vpar[0] = vmod*cos_p*B[0]/Bmod;
	vpar[1] = vmod*cos_p*B[1]/Bmod;
	vpar[2] = vmod*cos_p*B[2]/Bmod;

	//Perpendicular
	if(sin_p < 1E-10){
		double Ran0 = 2.0*PI*Ran_EC[1];
		vperp[0] = sin(Ran0)*vpar[2] - 0.0*vpar[1];
		vperp[1] = 0.0*vpar[0]       - cos(Ran0)*vpar[2];
		vperp[2] = cos(Ran0)*vpar[1] - sin(Ran0)*vpar[0];
	}else{
		vperp[0] = (*He).v[0] - vpar[0];
		vperp[1] = (*He).v[1] - vpar[1];
		vperp[2] = (*He).v[2] - vpar[2];
	}
	vperp_mod = sqrt(vperp[0]*vperp[0] + vperp[1]*vperp[1] + vperp[2]*vperp[2]);
	// ------------------------------------------------------------------------------

	double Ran0_p;
	double Ran0_E;

	Ran0_p = Ran_EC[2];
	if(Ran0_p > 0.5){
		Ran0_p = 1.0;
	}else{
		Ran0_p = -1.0;
	}
	Ran_gauss(&Ran_EC[2], 1.0);
	Ran0_E = Ran_EC[3];

	//Control:
	//printf("%.5e \t %.5e \t %.5e \n", E_au*27.2E-3, cos_p, sin_p);

	// Upgrade
	coll_freq(&v_col[0], E_au, Qa, r, th, z, &Ran_EC[0]);
	//printf("%.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e\n", v_col[0], v_col[1], v_col[2], v_col[3], v_col[4], v_col[5]);

	double F0_p, F0_E, G0_p, G0_E;
	F0_p = -(v_col[0]+v_col[1])*cos_p;
	F0_E = -(v_col[2]+v_col[3])*E_au;

	G0_p = sqrt( sin_p*sin_p*(v_col[0]+v_col[1]) );
	G0_E = 2.0*sqrt( E_au*(v_col[4]*T_e + v_col[5]*T_i) );

	cos_p = cos_p + Dt_au*F0_p + Ran0_p*sqrt(Dt_au)*G0_p;
	E_au = E_au + Dt_au*F0_E + Ran0_E*sqrt(Dt_au)*G0_E;
	if(cos_p < -1.0){
		cos_p = -1.0;
	}else if(cos_p > 1.0){
		cos_p = 1.0;
	}
	//printf("%.5e \t %.5e \n", cos_p, E_au);

	//final velocity (program's unit)
	vmod = sqrt(2.0*fabs(E_au)/Ma)*1.0/(137.0*v0_c);
	sin_p = sqrt(1.0 - cos_p*cos_p);



	//Final update;
	vpar[0] = vmod*cos_p*B[0]/Bmod;
	vpar[1] = vmod*cos_p*B[1]/Bmod;
	vpar[2] = vmod*cos_p*B[2]/Bmod;

	//printf("%.5e \t %.5e \t %.5e \n", vpar[0], vpar[1], vpar[2]);

	vperp[0] = vmod*sin_p*vperp[0]/vperp_mod;
	vperp[1] = vmod*sin_p*vperp[1]/vperp_mod;
	vperp[2] = vmod*sin_p*vperp[2]/vperp_mod;

	(*He).v[0] = vpar[0] + vperp[0];
	(*He).v[1] = vpar[1] + vperp[1];
	(*He).v[2] = vpar[2] + vperp[2];

	(*He).E_keV = dEp_MeV*vmod*vmod*1000.0;

	//printf("Right after the Collision: v=%f\t", (*He).v[0]);
}



/*
__device__ void Elastic_collisions_PE_MC_KP2_ito (struct Part *He, double Dt, double * Ran_EC){
	/* pitch - energy - Monte Carlo implementation* /

	if((*He).q == 0) return;
	double r = (*He).r[0];
	double th = (*He).r[1];
	double z = (*He).r[2];
	double Qa = (double)(*He).q;

	double v_col[6]; 	//collision frequencies

	//conversions
	double Dt_au = Dt*ta*c_cgs/(137.0*a0_cgs);
	double v0_c;	// = v0/c
	double Ma = 1836.2*mu;						// Projectile mass
	v0_c = sqrt(2.0*(dEp_MeV/27.2E-6)/Ma)/137.0;

	double vmod = sqrt( (*He).v[0]*(*He).v[0] + (*He).v[1]*(*He).v[1] + (*He).v[2]*(*He).v[2] );
	double v_au = vmod*v0_c*137.0;

	double E_au = 0.5*Ma*v_au*v_au;

	double T_e = Te( r, th, z )/27.2E-3;
	double T_i = T_e;

	double B[3], E[3], Bmod;
	magnetic_field(&B[0], &E[0], (*He).r[0], (*He).r[1], (*He).r[2], (*He).time);
	Bmod = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);


	//pitch inicial
	double cos_p = ((*He).v[0]*B[0] + (*He).v[1]*B[1] + (*He).v[2]*B[2])/(Bmod*vmod);
	double sin_p = sqrt(1.0 - cos_p*cos_p);

	// Parallel and Perpendicular vectors (respecto de B) ---------------------------
	double vpar[3];
	double vperp[3], vperp_mod;


	vpar[0] = vmod*cos_p*B[0]/Bmod;
	vpar[1] = vmod*cos_p*B[1]/Bmod;
	vpar[2] = vmod*cos_p*B[2]/Bmod;

	// la vel. perp no necesita estar normalizada.
	if(sin_p < 1E-10){
		double Ran0 = 2.0*PI*Ran_EC[0];
		vperp[0] = sin(Ran0)*vpar[2] - 0.0*vpar[1];
		vperp[1] = 0.0*vpar[0]       - cos(Ran0)*vpar[2];
		vperp[2] = cos(Ran0)*vpar[1] - sin(Ran0)*vpar[0];
	}else{
		vperp[0] = (*He).v[0] - vpar[0];
		vperp[1] = (*He).v[1] - vpar[1];
		vperp[2] = (*He).v[2] - vpar[2];
	}

	vperp_mod = sqrt(vperp[0]*vperp[0] + vperp[1]*vperp[1] + vperp[2]*vperp[2]);
	// ------------------------------------------------------------------------------

	double Ran0_p, Ran1_p;
	double Ran0_E, Ran1_E;

	Ran0_p = Ran_EC[0];
	if(Ran0_p > 0.5){
		Ran0_p = 1.0;
	}else{
		Ran0_p = -1.0;
	}

	Ran1_p = Ran_EC[1];
	if(Ran1_p > 0.5){
		Ran1_p = 1.0;
	}else{
		Ran1_p = -1.0;
	}

	Ran0_E = Ran_EC[2];
	Ran1_E = Ran_EC[3];

	//Control:
	//printf("%.5e \t %.5e \t %.5e \n", E_au*27.2E-3, cos_p, sin_p);

	// Etapa 0:
	double x0_p = cos_p;
	double x0_E = E_au;
	coll_freq(&v_col[0], x0_E, Qa, r, th, z, &Ran_EC[0]);

	double F0_p, F0_E, G0_p, G0_E;
	F0_p = -(v_col[0]+v_col[1])*x0_p;
	F0_E = -(v_col[2]+v_col[3])*x0_E;

	G0_p = sqrt( (1.0 - x0_p*x0_p)*(v_col[0]+v_col[1]) );
	G0_E = 2.0*sqrt( x0_E*(v_col[4]*T_e + v_col[5]*T_i) );

	// Etapa 1:
	double x1_p = x0_p + Dt_au*F0_p + Ran0_p*sqrt(Dt_au)*G0_p;
	double x1_E = x0_E + Dt_au*F0_E + Ran0_E*sqrt(Dt_au)*G0_E;
	coll_freq(&v_col[0], x1_E, Qa, r, th, z, &Ran_EC[0]);

	double F1_p, F1_E;
	F1_p = -(v_col[0]+v_col[1])*x1_p;
	F1_E = -(v_col[2]+v_col[3])*x1_E;

	// Etapa 2:
	double x2_p = x0_p + Ran1_p*sqrt(0.5*Dt_au)*G0_p;
	double x2_E = x0_E + Ran1_E*sqrt(0.5*Dt_au)*G0_E;
	coll_freq(&v_col[0], x2_E, Qa, r, th, z, &Ran_EC[0]);

	if(x2_p > 1.0 ) x2_p = 1.0;
	if(x2_p < -1.0 ) x2_p = -1.0;

	double G2_p, G2_E;
	G2_p = sqrt( (1.0 - x2_p*x2_p)*(v_col[0]+v_col[1]) );
	G2_E = 2.0*sqrt( x2_E*(v_col[4]*T_e + v_col[5]*T_i) );

	// Etapa 3:
	double x3_p = x0_p + Dt_au*F0_p + Ran1_p*sqrt(0.5*Dt_au)*G0_p;
	double x3_E = x0_E + Dt_au*F0_E + Ran1_E*sqrt(0.5*Dt_au)*G0_E;
	coll_freq(&v_col[0], x3_E, Qa, r, th, z, &Ran_EC[0]);

	double G3_p, G3_E;
	G3_p = sqrt( (1.0 - x3_p*x3_p)*(v_col[0]+v_col[1]) );
	G3_E = 2.0*sqrt( x3_E*(v_col[4]*T_e + v_col[5]*T_i) );

	//Actualizacion:
	cos_p = x0_p + 0.5*Dt_au*(F0_p + F1_p) + 0.5*Ran0_p*sqrt(Dt_au)*(G2_p + G3_p);

	if(cos_p > 1.0 ) cos_p = 1.0;
	if(cos_p < -1.0 ) cos_p = -1.0;

	E_au = x0_E + 0.5*Dt_au*(F0_E + F1_E) + 0.5*Ran0_E*sqrt(Dt_au)*(G2_E + G3_E);

	//printf("%.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \n", E_au*27.2E-3, cos_p, G0_p*sqrt(Dt_au), x2_p, G2_p*sqrt(Dt_au), G3_p*sqrt(Dt_au));

	//final velocity (program's unit)
	vmod = sqrt(2.0*fabs(E_au)/Ma)*1.0/(137.0*v0_c);
	sin_p = sqrt(1.0 - cos_p*cos_p);



	//Final update;
	vpar[0] = vmod*cos_p*B[0]/Bmod;
	vpar[1] = vmod*cos_p*B[1]/Bmod;
	vpar[2] = vmod*cos_p*B[2]/Bmod;

	vperp[0] = vmod*sin_p*vperp[0]/vperp_mod;
	vperp[1] = vmod*sin_p*vperp[1]/vperp_mod;
	vperp[2] = vmod*sin_p*vperp[2]/vperp_mod;

	(*He).v[0] = vpar[0] + vperp[0];
	(*He).v[1] = vpar[1] + vperp[1];
	(*He).v[2] = vpar[2] + vperp[2];

	(*He).E_keV = dEp_MeV*vmod*vmod*1000.0;
}
*/



__device__ void Elastic_collisions_SV_euler (struct Part *He, double Dt, double * Ran_EC){
	//printf("Inside 'Elastic_collisions_SV_euler': v=%f\t", (*He).v[0]);
	
	if((*He).q == 0) return;  // si es neutro no es colision coulombiana
	double r = (*He).r[0];
	double th = (*He).r[1];
	double z = (*He).r[2];
	double Qa = (double)(*He).q;  // carga de partícula prueba

	double v_col[6]; 	//collision frequencies

	//conversions
	double Dt_au = Dt*ta*c_cgs/(137.0*a0_cgs);  // Atomic units???
	double Ma = 1836.2*mu;						// Projectile mass
	double v0_c = sqrt(2.0*(dEp_MeV/27.2E-6)/Ma)/137.0; // = v0/c

	double vmod = sqrt( (*He).v[0]*(*He).v[0] + (*He).v[1]*(*He).v[1] + (*He).v[2]*(*He).v[2] );
	// hasta acá llega bien
	double v_au = vmod*v0_c*137.0;

	double E_au = 0.5*Ma*v_au*v_au;  // E_au=2941.176253

	//parallel and perpendicular vectors
	double v[3],w[3],u[3];
	double wmod;
	v[0] = (*He).v[0]/vmod;
	v[1] = (*He).v[1]/vmod;
	v[2] = (*He).v[2]/vmod;

	// w = v x (vr,0,vz) / |v x (vr,0,vz)|.
	if(v[1] == 0.0){
		w[0] = 0.0; w[1] = 1.0; w[2] = 0.0;
	} else{
		w[0] = v[1]*v[2];
		w[1] = 0.0;
		w[2] = - v[1]*v[0];
	}
	wmod = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
	w[0] = w[0]/wmod;
	w[1] = w[1]/wmod;
	w[2] = w[2]/wmod;

	// u = v x w / |v x w|
	u[0] = v[1]*w[2] - v[2]*w[1];  // u[0]=0.107007
	u[1] = v[2]*w[0] - v[0]*w[2];  //  u[1]=0.847216 
	u[2] = v[0]*w[1] - v[1]*w[0];  // u[2]=-0.520360


	coll_freq_par_perp(&v_col[0], E_au, Qa, r, th, z, &Ran_EC[0]);

	Ran_gauss(&Ran_EC[0], 1.0);
	Ran_gauss(&Ran_EC[2], 1.0);
	double Ran_1 = Ran_EC[2];
	double Ran_2 = Ran_EC[3];
	double Ran_3 = Ran_EC[1];

	//Upgrade
	double Dv_sl = -(v_col[0]+v_col[1])*Dt_au*vmod; // delta de fricción,  Dv_sl=4308688739221710.50
	double Dv_par =  sqrt((v_col[2]+v_col[3])*Dt_au)*vmod; // nan
	double Dv_perp = sqrt(0.5*(v_col[4]+v_col[5])*Dt_au)*vmod;  // nan


	(*He).v[0] = (*He).v[0] + (Dv_sl + Ran_1*Dv_par)*v[0] + Dv_perp*(Ran_2*w[0] + Ran_3*u[0]);  // NAn
	(*He).v[1] = (*He).v[1] + (Dv_sl + Ran_1*Dv_par)*v[1] + Dv_perp*(Ran_2*w[1] + Ran_3*u[1]);  //nan
	(*He).v[2] = (*He).v[2] + (Dv_sl + Ran_1*Dv_par)*v[2] + Dv_perp*(Ran_2*w[2] + Ran_3*u[2]);  // nan

	vmod = sqrt( (*He).v[0]*(*He).v[0] + (*He).v[1]*(*He).v[1] + (*He).v[2]*(*He).v[2] );  // = NaN
	//printf("Inside 'Elastic_collisions_SV_euler': vmod=%f\t", vmod);

	(*He).E_keV = dEp_MeV*vmod*vmod*1000.0;
	// printf("Energy after the Collision: E=%f\t", (*He).E_keV);

	// 2nd order energy correction -----------
	/*double v_2 = sqrt(vmod*vmod - Dv_sl*Dv_sl);
	(*He).v[0] = (*He).v[0]/vmod*v_2;
	(*He).v[1] = (*He).v[1]/vmod*v_2;
	(*He).v[2] = (*He).v[2]/vmod*v_2;
	(*He).E_keV = dEp_MeV*v_2*v_2*1000.0;*/
	// ----------------------------------------
}





__device__ void coll_freq(double *nu, double Eau, double Qa, double r, double th, double z,  double * Ran_EC){

	// Preliminary conversion.
	double v_au; 							// Projectile velocity
	double T_e, T_i; 						// Plasma Temperature
	double vth_e, vth_i, vs_e, vs_i;					// thermal velocities
	double wp, wp_i; 							// plasma frequency
	double xe, xi;							// v_au/vs_
	double LnAe, LnAi;						// Coulomb logarithm (e- , ions)
	double Gam_e, Gam_i;						// Gamma Factor
	double bt, biCL, beQM;						// Impact parameters (top, bottom)
	double Ge_par, Ge_perp, Gi_par, Gi_perp;			// Chandrasekar functions.
	double phi1e, phi1i;						// erf'(x) (e- , ions).
	double Mi;
	//if(Ran_EC[0]<0.5){
		Mi = 1836.2*mu;					// Target ion mass
	//}else{
		//Mi = 1836.2*mu_T;
	//}
	double Ma = 1836.2*mu;						// Projectile mass
	double ne_au, ni_au;

	double m_r = Mi*Ma/(Mi+Ma);

	v_au = sqrt(2.0*Eau/Ma); //v_au*v0_c*137.0;

	//Plasma parameters
	//functions:
	T_e = Te( r, th, z )/27.2E-3;
	ne_au = n_ei(r,th,z)*a0_cgs*a0_cgs*a0_cgs*1.0E14;
	ni_au = ne_au;

	T_i = T_e;
	vth_e = sqrt(T_e);
	vth_i = sqrt(T_i/Mi);
	vs_e = sqrt(2.0*T_e);
	vs_i = sqrt(2.0*T_i/Mi);

	wp = sqrt(4.0*PI*ne_au);
	wp_i = sqrt(4.0*PI*ni_au*Zb*Zb/Mi);

	// Coulomb Logarithm
	// Depende de la vel.(energia) pero desprecio esta variacion en
	// las etapas de integracion.
	bt = 1.0/sqrt(wp*wp/(vth_e*vth_e + v_au*v_au) + wp_i*wp_i/(vth_i*vth_i + v_au*v_au));	// bmax: an interpolation...
	biCL = Qa/m_r/(vs_i*vs_i + v_au*v_au);				// ions: classical limit (Bohr)
	beQM = 1.0/sqrt(vs_e*vs_e + v_au*v_au);							// elec: quantum limit (Bethe)

	LnAe = log(bt/beQM);
	LnAi = log(bt/biCL);

	//Gamma Factors
	Gam_e = 4.0*PI*Qa*Qa*LnAe/(Ma*Ma);
	Gam_i = 4.0*PI*Qa*Qa*Zb*Zb*LnAi/(Ma*Ma);

	// G functions
	xe = v_au/vs_e;
	xi = v_au/vs_i;
	phi1e = 2.0/sqrt(PI)*exp(-xe*xe);
	phi1i = 2.0/sqrt(PI)*exp(-xi*xi);

	Ge_par = (erf(xe) - xe*phi1e)/(2.0*xe*xe);
	Gi_par = (erf(xi) - xi*phi1i)/(2.0*xi*xi);

	Ge_perp = erf(xe) - Ge_par;
	Gi_perp = erf(xi) - Gi_par;


	// Frequencies calculations
	double v_E_e, v_E_i;
	v_E_e = 2.0*Gam_e*ne_au/(vs_e*vs_e*vs_e)*Ma*Ge_par/xe;
	v_E_i = 2.0*Gam_i*ni_au/(vs_i*vs_i*vs_i)*Ma/Mi*Gi_par/xi;


	nu[0] = Gam_e*ne_au/(v_au*v_au*v_au)*Ge_perp; //v_p_e
	nu[1] = Gam_i*ni_au/(v_au*v_au*v_au)*Gi_perp; //v_p_i

	nu[2] = 2.0*v_E_e*(1.0 - 0.5*xe*phi1e/Ge_par*T_e/Eau); //veff_e
	nu[3] = 2.0*v_E_i*(1.0 - 0.5*xi*phi1i/Gi_par*T_i/Eau); //veff_i

	nu[4] = v_E_e; // used in energy diffusion
	nu[5] = v_E_i;

}




__device__ void coll_freq_par_perp(double *nu, double Eau, double Qa, double r, double th, double z, double * Ran_EC){

	
	// Preliminary conversion.
	double v_au; 							// Projectile velocity
	double T_e, T_i; 						// Plasma Temperature
	double vth_e, vth_i, vs_e, vs_i;					// thermal velocities
	double wp, wp_i; 							// plasma frequency
	double xe, xi;							// v_au/vs_
	double LnAe, LnAi;						// Coulomb logarithm (e- , ions)
	double Gam_e, Gam_i;						// Gamma Factor
	double bt, biCL, beQM;						// Impact parameters (top, bottom)
	double Ge_par, Ge_perp, Gi_par, Gi_perp;			// Chandrasekar functions.
	double phi1e, phi1i;						// erf'(x) (e- , ions).
	double Mi;
	//if(Ran_EC[0]<0.5){
		Mi = 1836.2*mu;					// Target ion mass
	//}else{
		//Mi = 1836.2*mu_T;
	//}
	double Ma = 1836.2*mu;						// Projectile mass
	double ne_au, ni_au;

	double m_r = Mi*Ma/(Mi+Ma);

	v_au = sqrt(2.0*Eau/Ma); //v_au*v0_c*137.0;

	//Plasma parameters
	//functions:
	T_e = Te( r, th, z )/27.2E-3;
	ne_au = n_ei(r,th,z)*a0_cgs*a0_cgs*a0_cgs*1.0E14;
	//T_e = Te( r, th, z );  // T_e=0.000000
	//ne_au = n_ei(r,th,z);  // normalized, ne_au=0.000000 
	ni_au = ne_au;
	//printf("Inside 'coll_freq_par_perp': T_e=%f\t", T_e);
	//printf("Inside 'coll_freq_par_perp': ne_au=%f\t", ne_au);

	T_i = T_e;
	vth_e = sqrt(T_e);
	vth_i = sqrt(T_i/Mi);
	vs_e = sqrt(2.0*T_e);  //vs_e=0.000000 
	vs_i = sqrt(2.0*T_i/Mi);  // vs_i=0.000000
	

	wp = sqrt(4.0*PI*ne_au);
	wp_i = sqrt(4.0*PI*ni_au*Zb*Zb/Mi);

	// Coulomb Logarithm
	// Depende de la vel.(energia) pero desprecio esta variacion en
	// las etapas de integracion.
	bt = 1.0/sqrt(wp*wp/(vth_e*vth_e + v_au*v_au) + wp_i*wp_i/(vth_i*vth_i + v_au*v_au));	// bmax: an interpolation...
	biCL = Qa/m_r/(vs_i*vs_i + v_au*v_au);				// ions: classical limit (Bohr)
	beQM = 1.0/sqrt(vs_e*vs_e + v_au*v_au);							// elec: quantum limit (Bethe)

	LnAe = log(bt/beQM);  // LnAe=-24.243050
	LnAi = log(bt/biCL);

	

	//Gamma Factors
	Gam_e = 4.0*PI*Qa*Qa*LnAe/(Ma*Ma);
	Gam_i = 4.0*PI*Qa*Qa*Zb*Zb*LnAi/(Ma*Ma);

	// G functions
	xe = v_au/vs_e;  // xe=238074231865.732208
	xi = v_au/vs_i;  // xi=14427377023203.166016 
	phi1e = 2.0/sqrt(PI)*exp(-xe*xe);
	phi1i = 2.0/sqrt(PI)*exp(-xi*xi);

	Ge_par = (erf(xe) - xe*phi1e)/(2.0*xe*xe);
	Gi_par = (erf(xi) - xi*phi1i)/(2.0*xi*xi);

	Ge_perp = erf(xe) - Ge_par;  // Ge_perp=1.000000
	Gi_perp = erf(xi) - Gi_par;  // Gi_perp=1.000000 

	// Frequencies calculations
	nu[0] = 2.0*Gam_e*ne_au/(vs_e*vs_e*vs_e)*(1.0+Ma)*Ge_par/xe;	//nu_s_e
	nu[1] = 2.0*Gam_i*ni_au/(vs_i*vs_i*vs_i)*(1.0+Ma/Mi)*Gi_par/xi;	//nu_s_i


	nu[2] = 2.0*Gam_e*ne_au/(v_au*v_au*v_au)*Ge_par; 		//nu_par_e
	nu[3] = 2.0*Gam_i*ni_au/(v_au*v_au*v_au)*Gi_par; 		//nu_par_i

	nu[4] = 2.0*Gam_e*ne_au/(v_au*v_au*v_au)*Ge_perp; 		//nu_perp_i
	nu[5] = 2.0*Gam_i*ni_au/(v_au*v_au*v_au)*Gi_perp; 		//nu_perp_i


}
