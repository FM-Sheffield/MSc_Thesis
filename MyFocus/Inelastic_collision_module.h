/* ********** Atomic processes ***************/


// ============== Qv matrices =================================
// Devuelve la frec. colision (1/seg) para una densidad n_e = 10^{14}
// Es decir, multiplicar por la fraccion de la densidad 
// Inicializadas con la fn init_matrices!
// Los datos de las matrices son calculados con data de las secciones eficaces / <sigma.v>=sigma(v)*v

// HOST ***************
#ifdef Z_2
// alpha particles
double hCE_He2_H[51][2];
double hCE_He2_H2[50][2];
double hCE_He2_He[100][2];
double hCE_He1_H[51][2];
double hCE_He1_H2[51][2];
double hCE_He1_He[101][2];
double hCE_He1_He1[13][2];
double hCE_He2_He1[15][2];

double hEII1[21][8];				// Tmin = 0 eV, Tmax = 20 keV
double hER_He1_D1[14][7];			// Tmin = 0 keV, Tmax = 20 keV

double hEII0[16][6];
double hER_He0_D1[16][6];
double hCE_He0_H2_He1[18][2];
#endif

#ifdef Z_1
// Deuterium particles
double hVe_exc_12[11][7];  // exitación con... electrón a nivel más alto?
double hVe_exc_13[11][7];
double hVe_exc_23[11][7];

double hVe_ion_1[11][7];   // ionización con... electrón 1, 2, 3? capáz es ionización con deuterio excitado
double hVe_ion_2[11][7];
double hVe_ion_3[11][7];

double hVi_exc_12[11][4];  // exitación con... ion (deuterio?) 1, 2, 3? 
double hVi_exc_13[11][4];
double hVi_exc_23[11][4];

double hVi_ion_1[11][4];
double hVi_ion_2[11][4];
double hVi_ion_3[11][4];

double hVi_ec_1[11][4];
double hVi_ec_2[11][4];
double hVi_ec_3[11][5];
#endif

// DEVICE **************
#ifdef Z_2
__constant__ __device__ double CE_He2_H[51][2];  // captura electrónica
__constant__ __device__ double CE_He2_H2[50][2];
__constant__ __device__ double CE_He2_He[100][2];
__constant__ __device__ double CE_He2_He1[15][2];

__constant__ __device__ double CE_He1_H[51][2];
__constant__ __device__ double CE_He1_H2[51][2];
__constant__ __device__ double CE_He1_He[101][2];
__constant__ __device__ double CE_He1_He1[13][2];

__constant__ __device__ double EII1[21][8];                             // Tmin = 0 eV, Tmax = 20 keV
__constant__ __device__ double ER_He1_D1[14][7];                       // Tmin = 0 keV, Tmax = 20 keV

__constant__ __device__ double EII0[16][6];		// ionización por impacto de electrones (EII)
__constant__ __device__ double ER_He0_D1[16][6];
__constant__ __device__ double CE_He0_H2_He1[18][2];
#endif

#ifdef Z_1
__constant__ __device__ double Ve_exc_12[11][7];  // Excitacion por impacto de electrones.
__constant__ __device__ double Ve_exc_13[11][7];
__constant__ __device__ double Ve_exc_23[11][7];

__constant__ __device__ double Ve_ion_1[11][7];  // ionizacion por impacto de electrones.
__constant__ __device__ double Ve_ion_2[11][7];
__constant__ __device__ double Ve_ion_3[11][7];

__constant__ __device__ double Vi_exc_12[11][4];  // Excitacion  con deuterones
__constant__ __device__ double Vi_exc_13[11][4];
__constant__ __device__ double Vi_exc_23[11][4];

__constant__ __device__ double Vi_ion_1[11][4];  //ionizacion con deuterones
__constant__ __device__ double Vi_ion_2[11][4];
__constant__ __device__ double Vi_ion_3[11][4];

__constant__ __device__ double Vi_ec_1[11][4];
__constant__ __device__ double Vi_ec_2[11][4];  //intercambio de carga con deuterones
__constant__ __device__ double Vi_ec_3[11][5];
#endif
//=================================================*/


void load_atomic_processes(void); // Load <qv> files in matrices.
__device__ void Inelastic_collisions (struct Part *He, double Dt_seg, int * i, long init, int tid);
__device__ void Inelastic_collisions_He (struct Part *He, double Dt_seg, double Ran);
__device__ void Inelastic_collisions_D (struct Part *He, double Dt_seg, double Ran);


/* ******** interpol_F *************************
 * for CE. Gives an interpolation of a "nrow x 2" matrix.
 * The targets are cold. Plasma temp = 0.*/
__host__ __device__ double interpol_F(double EkeV, 		// value
				  double *pQv, 		// vector
				  int nrow, 		// dim vector
				  int ncol);		// equal = 2.

/* ******** interpol_T *************************
 * for EII y PII. Gives an interpolation of a matrix.*/
__host__ __device__ double interpol_T(double EkeV, 		// row value
				  double T, 		// column value
				  double *pQv, 		// Pointer to first matrix element.
				  int nrow, 		// number of rows
				  int ncol);		// number of columns.


__device__ void Inelastic_collisions (struct Part *He, double Dt_seg, int * i, long init, int tid){

	//Random numbers initialization ------
	philox4x32_ctr_t   c={{}};
	philox4x32_key_t   k={{}};
	philox4x32_ctr_t   r;

	k.v[0] = tid; 		// thread id
	k.v[1] = init;		// global seed
	c.v[0] = *i;		// global counter (iteration index)
	*i = *i + 1;
	//------------------------------------
	r = philox4x32(c, k);

	double Ran = (double) u01_open_open_32_53(r.v[0]);
	//printf("%f \n", Ran);

#ifdef Z_2
		Inelastic_collisions_He (He, Dt_seg, Ran);
#endif

#ifdef Z_1
		Inelastic_collisions_D (He, Dt_seg, Ran);
#endif
}



#ifdef Z_1
__device__ void Inelastic_collisions_D (struct Part *He, double Dt_seg, double Ran){ 
	//printf("Posible col \n");
	double P_1, P_2, P_3;
	// double r = (*He).r[0];
	// double x = sqrt( (r-R0)*(r-R0) + (*He).r[2]*(*He).r[2] );

	//double x = (*He).r[0] - R0;
	double TkeV = Te( (*He).r[0], (*He).r[1], (*He).r[2] );
	//printf("T_kev: %f\t", TkeV);
	double ncm3 = n_ei((*He).r[0], (*He).r[1], (*He).r[2] );
	//de-excitation times
	double Tau21 = 1.0/4.699E8;  // del electrón 
	double Tau31 = 1.0/5.575E7;
	double Tau32 = 1.0/4.410E7;

	if( (*He).q == 0 ){		// deuterio neutro
		//printf("Entro a q=0\n");
		if((*He).n == 1){
			//printf("Entro a q=0, n=1\n");
			
			//excitation 1->2
			P_1 = ncm3*interpol_T( (*He).E_keV,TkeV, &Ve_exc_12[0][0],11,7)*Dt_seg;  // frec de col * tiempo 
			P_1 = P_1 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_exc_12[0][0],11,4)*Dt_seg;
			//excitation 1->3
			P_2  = ncm3*interpol_T( (*He).E_keV,TkeV, &Ve_exc_13[0][0],11,7)*Dt_seg;
			P_2  = P_2 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_exc_13[0][0],11,4)*Dt_seg;
			//ionization + ec 1->free
			P_3  = ncm3*interpol_T( (*He).E_keV,TkeV, &Ve_ion_1[0][0],11,7)*Dt_seg;
			P_3  = P_3 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_ion_1[0][0],11,4)*Dt_seg;
			P_3  = P_3 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_ec_1[0][0],11,4)*Dt_seg;
			// P1=0.014624        , P2=0.002880   , P3=0.031756; test rápido en mi equilibrio - D positiva
			// printf("n=1, P1=%f\t, P2=%f\t, P3=%f\n", P_1, P_2, P_3);  // todas las P's << 1 para que valga el modelo -> fija Nic
			
			if( Ran < P_1 ){
				(*He).n = 2;
				(*He).timeAt = 0.0;
			}else if(Ran > P_1 && Ran < (P_1 + P_2)){
				(*He).n = 3;
				(*He).timeAt = 0.0;
			}else if(Ran > (P_1 + P_2) && Ran < (P_1 + P_2 + P_3)){
				(*He).n = 0;
				(*He).q = 1;
				(*He).timeAt = 0.0;
				//printf("ionizada n=1\n");
			}
			
		}else if((*He).n == 2){
			//printf("Entro a q=0, n=2\n");
			//excitation 2->3
			P_1 = ncm3*interpol_T( (*He).E_keV,TkeV, &Ve_exc_23[0][0],11,7)*Dt_seg;
			P_1 = P_1 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_exc_23[0][0],11,7)*Dt_seg;
			//ionization + ec 2->free
			P_2 = ncm3*interpol_T( (*He).E_keV,TkeV, &Ve_ion_2[0][0],11,7)*Dt_seg;
			P_2 = P_2 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_ion_2[0][0],11,4)*Dt_seg;
			P_2 = P_2 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_ec_2[0][0],11,4)*Dt_seg;
			//de-excitation 2->1
			P_3 = 1.0 - exp(-(*He).timeAt/Tau21);
			//P_3 = 0.0;
			if( Ran < P_1 ){
				(*He).n = 3;
				(*He).timeAt = 0.0;
			}else if(Ran > P_1 && Ran < (P_1 + P_2)){
				(*He).n = 0;
				(*He).q = 1;
				(*He).timeAt = 0.0;
				//printf("ionizada n=2\n");
			}else if(Ran > (P_1 + P_2) && Ran < (P_1 + P_2 + P_3)){
				(*He).n = 1;
				(*He).timeAt = 0.0;
			}
		}else if((*He).n == 3){
			//printf("Entro a q=0, n=3\n");
			//de-excitation 3->1
			P_1 = 1.0 - exp(-(*He).timeAt/Tau31);
			//P_1 = 0.0;			
			//de-excitation 3->2
			P_2 = 1.0 - exp(-(*He).timeAt/Tau32);
			//P_2 = 0.0;
			//ionization + ce 3->free
			P_3 = ncm3*interpol_T( (*He).E_keV,TkeV, &Ve_ion_3[0][0],11,7)*Dt_seg;
			P_3 = P_3 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_ion_3[0][0],11,4)*Dt_seg;
			P_3 = P_3 + ncm3*interpol_T( (*He).E_keV,TkeV, &Vi_ec_3[0][0],11,5)*Dt_seg;
			if( Ran < P_1 ){
				(*He).n = 1;
				(*He).timeAt = 0.0;
			}else if(Ran > P_1 && Ran < (P_1 + P_2)){
				(*He).n = 2;
				(*He).timeAt = 0.0;
			}else if(Ran > (P_1 + P_2) && Ran < (P_1 + P_2 + P_3)){
				(*He).n = 0;
				(*He).q = 1;
				(*He).timeAt = 0.0;
				//printf("ionizada n=3\n");
			}
		}
	}else if( (*He).q == 1 ){
		//Aca podrían ir neutralizaciones. Fijarse Muscatello: Hay bastante Carbono!
		//printf("Entro a q=1\n");
		return;
	}
}
#endif



#ifdef Z_2
__device__ void Inelastic_collisions_He (struct Part *He, double Dt_seg, double Ran){
	//Ran = 0.0;

	double P_CE0, P_CE1, P_CE2,P_CE3;
	double P_EII, P_PII_CE;
	double r =  (*He).r[0];
	double th = (*He).r[1];
	double z =  (*He).r[2];
	
	double TkeV = Te(r,th,z);
	double ncm3 = n_ei(r,th,z);

	if( (*He).q == 2 ){		
		P_CE0 = nH*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He2_H[0][0],51,2)*Dt_seg;
		P_CE1 = nH2*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He2_H2[0][0],50,2)*Dt_seg;
		P_CE2 = nHe*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He2_He[0][0],100,2)*Dt_seg;
		P_CE3 = nHe1*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He2_He1[0][0],15,2)*Dt_seg;
		if( Ran < (P_CE0+P_CE1+P_CE2+P_CE3) ){
			(*He).q = (*He).q - 1;
		}
	} else if( (*He).q == 1 ){
		// La reacción He1 + He1 puede recombinar como ionizar
		P_CE0 = nH*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He1_H[0][0],51,2)*Dt_seg;
		P_CE1 = nH2*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He1_H2[0][0],51,2)*Dt_seg;
		P_CE2 = nHe*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He1_He[0][0],101,2)*Dt_seg;
		P_CE3 = 0.5*nHe1*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He1_He1[0][0],13,2)*Dt_seg;
		
		P_CE0 = P_CE0+P_CE1+P_CE2+P_CE3;
		
		P_EII = ncm3*interpol_T( (*He).E_keV, TkeV,&EII1[0][0],21,8)*Dt_seg;
		P_PII_CE = ncm3*interpol_T((*He).E_keV, TkeV,&ER_He1_D1[0][0],14,7)*Dt_seg;
		
		P_EII = P_EII + P_PII_CE + P_CE3;
		
		if( Ran < P_CE0 ){
			(*He).q = (*He).q - 1;
		}else if (Ran > P_CE0 && Ran < (P_CE0+P_EII)){
			(*He).q = (*He).q + 1;
		}
	} else if( (*He).q == 0 ){
		P_EII = ncm3*interpol_T( (*He).E_keV, TkeV,&EII0[0][0],16,6)*Dt_seg;
		P_PII_CE = P_EII + ncm3*interpol_T((*He).E_keV, TkeV,&ER_He0_D1[0][0],16,6)*Dt_seg;
		P_CE1 = P_PII_CE + nH2*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He0_H2_He1[0][0],18,2)*Dt_seg;
		P_CE2 = P_CE1 + nHe1*f_n(r,th,z)*interpol_F( (*He).E_keV,&CE_He1_He[0][0],101,2)*Dt_seg;
		if( Ran < P_CE2 ){
			(*He).q = (*He).q + 1;
		}
	}
}
#endif




__device__ double interpol_F(double EkeV, double *pQv, int nrow, int ncol){

	int i,j;	
	//double Qv[nrow][ncol];
	double Qvfinal;

	//Qv[i][j] = pQv[i*ncol + j];

	// Interpolacion lineal -----------------
	// Asignacion de indices.
	i = nrow;
	j = ncol - 1;
	do{
		i = i - 1;
	}while(EkeV<pQv[i*ncol + 0]);
	
	Qvfinal = pQv[i*ncol + j] + 
	         (pQv[(i+1)*ncol + j] - pQv[i*ncol + j])*(EkeV - pQv[i*ncol + 0])/(pQv[(i+1)*ncol + 0] - pQv[i*ncol + 0]);	
	// ------------------------------------
	
	return Qvfinal;
}
__device__ double interpol_T(double EkeV, double T, double *pQv, int nrow, int ncol) {
	/*
	(double EkeV, 		// row value
				  double T, 		// column value
				  double *pQv, 		// Pointer to first matrix element.
				  int nrow, 		// number of rows
				  int ncol);		// number of columns.
	*/


	// La matriz Qv tiene la Temperatura en la fila 0
	// y la Energia en la columna 0.
	//double Qv[nrow][ncol];
	double Qv1, Qv2;
	double Qvfinal;
	int i,j;	
	
	//Qv[i][j] = pQv[i*ncol + j];
	
	// Interpolacion lineal -----------------
		// Asignacion de indices.
		i = nrow-1;
		do{
			i = i - 1;
		}while(EkeV<pQv[i*ncol + 0]);

		j = ncol-1;
		do{
			j = j - 1;
		}while(T<pQv[0*ncol + j]);
		// 1er paso: interpolar cada columna:
		Qv1 = pQv[i*ncol + j] + (pQv[(i+1)*ncol + j] - pQv[i*ncol + j])*
		      (EkeV - pQv[i*ncol + 0])/(pQv[(i+1)*ncol + 0] - pQv[i*ncol + 0]);
		Qv2 = pQv[i*ncol + j+1] + (pQv[(i+1)*ncol + j+1] - pQv[i*ncol + j+1])
		      *(EkeV - pQv[i*ncol + 0])/(pQv[(i+1)*ncol + 0] - pQv[i*ncol + 0]);
		// 2do paso: Interpolar en filas:
		Qvfinal = Qv1 + (Qv2 - Qv1)*(T - pQv[0*ncol + j])/(pQv[0*ncol + j+1] - pQv[0*ncol + j]);
	
	// ------------------------------------
	
	return Qvfinal;
}







void load_atomic_processes(void) {

	int i,j;
	//Archivo de lectura 
	FILE *Datos_Qv;
	
	// ======= CARGO TODAS LAS MATRICES ===================

#ifdef Z_2
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He2_H_to_He1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<51;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf",&hCE_He2_H[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, CE_He2_H[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He2_H2_to_He1_single.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<50;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He2_H2[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He2_He_to_He1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<100;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He2_He[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He1_H_to_He0.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<51;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He1_H[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He1_H2_to_He0.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<51;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He1_H2[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He1_He_to_He0.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<101;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He1_He[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He2_He1_to_He1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<15;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He2_He1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He1_He1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<13;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He1_He1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/EII_He1_to_He2.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<21;i++){
		for(j=0;j<8;j++){
			fscanf(Datos_Qv, "%lf ",&hEII1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/ER_He1_D1_to_He2.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<14;i++){
		for(j=0;j<7;j++){
			fscanf(Datos_Qv, "%lf ",&hER_He1_D1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/EII_He0.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<16;i++){
		for(j=0;j<6;j++){
			fscanf(Datos_Qv, "%lf ",&hEII0[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/ER_He0_D1_to_He1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<16;i++){
		for(j=0;j<6;j++){
			fscanf(Datos_Qv, "%lf ",&hER_He0_D1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/alpha/CE_He0_H2_to_He1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<18;i++){
		for(j=0;j<2;j++){
			fscanf(Datos_Qv, "%lf ",&hCE_He0_H2_He1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
#endif




	// ------------------------------------------------

#ifdef Z_1

	Datos_Qv = fopen("Atomic_processes/deuterium/ve_exc_12.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<7;j++){
			fscanf(Datos_Qv, "%lf ",&hVe_exc_12[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/ve_exc_13.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<7;j++){
			fscanf(Datos_Qv, "%lf ",&hVe_exc_13[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/ve_exc_23.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<7;j++){
			fscanf(Datos_Qv, "%lf ",&hVe_exc_23[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------

	Datos_Qv = fopen("Atomic_processes/deuterium/ve_ion_1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<7;j++){
			fscanf(Datos_Qv, "%lf ",&hVe_ion_1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/ve_ion_2.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<7;j++){
			fscanf(Datos_Qv, "%lf ",&hVe_ion_2[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/ve_ion_3.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<7;j++){
			fscanf(Datos_Qv, "%lf ",&hVe_ion_3[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------


	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/vi_exc_12.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_exc_12[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/vi_exc_13.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_exc_13[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/vi_exc_23.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_exc_23[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------

	Datos_Qv = fopen("Atomic_processes/deuterium/vi_ion_1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_ion_1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/vi_ion_2.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_ion_2[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/vi_ion_3.dat","r");

	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.

	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_ion_3[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);




	// ------------------------------------------------

	Datos_Qv = fopen("Atomic_processes/deuterium/vi_ec_1.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_ec_1[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);
	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/vi_ec_2.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<4;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_ec_2[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
	Datos_Qv = fopen("Atomic_processes/deuterium/vi_ec_3.dat","r");
	if(Datos_Qv == NULL){
		printf("Error Datos_Qv");
		exit(1);}
	// Cargo en la matriz Qv[][] el archivo de datos.
	for(i=0;i<11;i++){
		for(j=0;j<5;j++){
			fscanf(Datos_Qv, "%lf ",&hVi_ec_3[i][j]);
			//printf("matriz: Qv[%d][%d]: %f \n ",i, j, Qv[i][j]);
		}	
	}
	fclose(Datos_Qv);

	// ------------------------------------------------
#endif



	// --------- COPY TO DEVICE --------------------------------
#ifdef Z_2
//Alpha particles	
        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He2_H, hCE_He2_H, 51*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He2_H2, hCE_He2_H2, 50*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He2_He, hCE_He2_He, 100*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He2_He1, hCE_He2_He1, 15*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));

        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He1_H, hCE_He1_H, 51*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He1_H2, hCE_He1_H2, 51*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He1_He, hCE_He1_He, 101*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol( CE_He1_He1, hCE_He1_He1, 13*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));


        HANDLE_ERROR(cudaMemcpyToSymbol( EII1, hEII1, 21*8*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol(ER_He1_D1, hER_He1_D1, 14*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
	HANDLE_ERROR(cudaMemcpyToSymbol( EII0, hEII0, 16*6*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
        HANDLE_ERROR(cudaMemcpyToSymbol(ER_He0_D1, hER_He0_D1, 16*6*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
	HANDLE_ERROR(cudaMemcpyToSymbol( CE_He0_H2_He1, hCE_He0_H2_He1, 18*2*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
#endif

#ifdef Z_1
//Haz de deuterio
       HANDLE_ERROR(cudaMemcpyToSymbol( Ve_exc_12, hVe_exc_12, 11*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Ve_exc_13, hVe_exc_13, 11*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Ve_exc_23, hVe_exc_23, 11*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));

       HANDLE_ERROR(cudaMemcpyToSymbol( Ve_ion_1, hVe_ion_1, 11*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Ve_ion_2, hVe_ion_2, 11*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Ve_ion_3, hVe_ion_3, 11*7*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));

       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_exc_12, hVi_exc_12, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_exc_13, hVi_exc_13, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_exc_23, hVi_exc_23, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));

       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_ion_1, hVi_ion_1, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_ion_2, hVi_ion_2, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_ion_3, hVi_ion_3, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));

       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_ec_1, hVi_ec_1, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_ec_2, hVi_ec_2, 11*4*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
       HANDLE_ERROR(cudaMemcpyToSymbol( Vi_ec_3, hVi_ec_3, 11*5*sizeof(double), size_t(0), cudaMemcpyHostToDevice ));
#endif


	checkCUDAError("copy GPU: failed \n");
	
}


















