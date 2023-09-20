//================================================================================================//
//------------------------------------------------------------------------------------------------//
//    MPH-WC : Moving Particle Hydrodynamics for Weakly Compressible  (explicit calculation)      //
//------------------------------------------------------------------------------------------------//
//    Developed by    : Masahiro Kondo                                                            //
//    Distributed in  : 2022                                                                      //
//    Lisence         : GPLv3                                                                     //
//    For instruction : see README                                                                //
//    For theory      : see the following references                                              //
//     [1] CPM 8 (2021) 69-86,       https://doi.org/10.1007/s40571-020-00313-w                   //
//     [2] JSCES Paper No.20210006,  https://doi.org/10.11421/jsces.2021.20210006                 //
//     [3] CPM 9 (2022) 265-276,     https://doi.org/10.1007/s40571-021-00408-y                   //
//     [4] CMAME 385 (2021) 114072,  https://doi.org/10.1016/j.cma.2021.114072                    //
//     [5] JSCES Paper No.20210016,  https://doi.org/10.11421/jsces.2021.20210016                 //
//    (Please cite the references above when you make a publication using this program)           //
//    Copyright (c) 2022                                                                          //
//    Masahiro Kondo & National Institute of Advanced Industrial Science and Technology (AIST)    //
//================================================================================================//


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <assert.h>
#include <omp.h>

#include "errorfunc.h"
#include "log.h"

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
#define TWO_DIMENSIONAL

#define DIM 3

// Property definition
#define TYPE_COUNT   4
#define FLUID_BEGIN  0
#define FLUID_END    2
#define WALL_BEGIN   2
#define WALL_END     4

#define  DEFAULT_LOG  "sample.log"
#define  DEFAULT_DATA "sample.data"
#define  DEFAULT_GRID "sample.grid"
#define  DEFAULT_PROF "sample%03d.prof"
#define  DEFAULT_VTK  "sample%03d.vtk"

// Calculation and Output
static double ParticleSpacing=0.0;
static double ParticleVolume=0.0;
static double OutputInterval=0.0;
static double OutputNext=0.0;
static double VtkOutputInterval=0.0;
static double VtkOutputNext=0.0;
static double EndTime=0.0;
static double Time=0.0;
static double Dt=1.0e100;
static double DomainMin[DIM];
static double DomainMax[DIM];
static double DomainWidth[DIM];
#pragma acc declare create(ParticleSpacing,ParticleVolume,Dt,DomainMin,DomainMax,DomainWidth)

#define Mod(x,w) ((x)-(w)*floor((x)/(w)))   // mod 

#define MAX_NEIGHBOR_COUNT 512
// Particle
static int ParticleCount;
static int *ParticleIndex;                // original particle id
static int *Property;                     // particle type
static double (*Mass);                    // mass
static double (*Position)[DIM];           // coordinate
static double (*Velocity)[DIM];           // momentum
static double (*Force)[DIM];              // total explicit force acting on the particle
static int (*NeighborCount);                  // [ParticleCount]
static long int (*NeighborPtr);           // [ParticleCount+1];
static int          (*NeighborInd);           // [ParticleCount x NeighborCount]
static long int   NeighborIndCount;
static double (*NeighborCalculatedPosition)[DIM];
static int    (*TmpIntScalar);                // [ParticleCount] to sort with cellId
static double (*TmpDoubleScalar);             // [ParticleCount]
static double (*TmpDoubleVector)[DIM];        // [ParticleCount]
#define MARGIN (0.1*ParticleSpacing)
#pragma acc declare create(ParticleCount,ParticleIndex,Property,Mass,Position,Velocity,Force,NeighborCalculatedPosition)
#pragma acc declare create(NeighborCount,NeighborPtr,NeighborInd,NeighborIndCount)
#pragma acc declare create(TmpIntScalar,TmpDoubleScalar,TmpDoubleVector)


// BackGroundCells
#ifdef TWO_DIMENSIONAL
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1])
#else
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1]*CellCount[2] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1]*CellCount[2] + ((iCZ)%CellCount[2]+CellCount[2])%CellCount[2])
#endif

static int PowerParticleCount;
static int ParticleCountPower;                   
static double CellWidth[DIM];
static int CellCount[DIM];
static int TotalCellCount = 0;
static int *CellFluidParticleBegin;  // beginning of fluid particles in the cell
static int *CellFluidParticleEnd;  // end of fluid particles in the cell
static int *CellWallParticleBegin;   // beginning of wall particles in the cell
static int *CellWallParticleEnd;   // end of wall particles in the cell
static int *CellIndex;  // [ParticleCountPower>>1]
static int *CellParticle;       // array of particle id in the cells) [ParticleCountPower>>1]
#pragma acc declare create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount,TotalCellCount)
#pragma acc declare create(CellFluidParticleBegin,CellFluidParticleEnd,CellWallParticleBegin,CellWallParticleEnd,CellIndex,CellParticle)

// Type
static double Density[TYPE_COUNT];
static double BulkModulus[TYPE_COUNT];
static double BulkViscosity[TYPE_COUNT];
static double ShearViscosity[TYPE_COUNT];
static double SurfaceTension[TYPE_COUNT];
static double CofA[TYPE_COUNT];   // coefficient for attractive pressure
static double CofK;               // coefficinet (ratio) for diffuse interface thickness normalized by ParticleSpacing
static double InteractionRatio[TYPE_COUNT][TYPE_COUNT];
#pragma acc declare create(Density,BulkModulus,BulkViscosity,ShearViscosity,SurfaceTension,CofA,CofK,InteractionRatio)


// Fluid
static int FluidParticleBegin;
static int FluidParticleEnd;
static double *DensityA;        // number density per unit volume for attractive pressure
static double (*GravityCenter)[DIM];
static double *PressureA;       // attractive pressure (surface tension)
static double *VolStrainP;        // number density per unit volume for base pressure
static double *DivergenceP;     // volumetric strainrate for pressure B
static double *PressureP;       // base pressure
static double *VirialPressureAtParticle; // VirialPressureInSingleParticleRegion
static double (*VirialStressAtParticle)[DIM][DIM];
static double *Mu;              // viscosity coefficient for shear
static double *Lambda;          // viscosity coefficient for bulk
static double *Kappa;           // bulk modulus
#pragma acc declare create(FluidParticleBegin,FluidParticleEnd)
#pragma acc declare create(DensityA,GravityCenter,PressureA,VolStrainP,DivergenceP,PressureP,VirialPressureAtParticle,VirialStressAtParticle,Mu,Lambda,Kappa)

static double Gravity[DIM] = {0.0,0.0,0.0};
#pragma acc declare create(Gravity)

// Wall
static int WallParticleBegin;
static int WallParticleEnd;
static double WallCenter[WALL_END][DIM];
static double WallVelocity[WALL_END][DIM];
static double WallOmega[WALL_END][DIM];
static double WallRotation[WALL_END][DIM][DIM];
#pragma acc declare create(WallParticleBegin,WallParticleEnd)
#pragma acc declare create(WallCenter,WallVelocity,WallOmega,WallRotation)

// proceedures
static void readDataFile(char *filename);
static void readGridFile(char *filename);
static void writeProfFile(char *filename);
static void writeVtkFile(char *filename);
static void initializeWeight( void );
static void initializeDomain( void );
static void initializeFluid( void );
static void initializeWall( void );

static void calculateConvection();
static void calculateWall();
static void calculatePeriodicBoundary();
static void resetForce();
static int neighborCalculation();
static void calculateCellParticle( void );
static void calculateNeighbor( void );
static void freeNeighbor( void );
static void calculateNeighbor();
static void calculatePhysicalCoefficients();
static void calculateDensityA();
static void calculatePressureA();
static void calculateGravityCenter();
static void calculateDiffuseInterface();
static void calculateDensityP();
static void calculateDivergenceP();
static void calculatePressureP();
static void calculateViscosityV();
static void calculateGravity();
static void calculateAcceleration();
static void calculateVirialPressureAtParticle();
static void calculateVirialStressAtParticle();


// dual kernel functions
static double RadiusRatioA;
static double RadiusRatioG;
static double RadiusRatioP;
static double RadiusRatioV;

static double MaxRadius = 0.0;
static double RadiusA = 0.0;
static double RadiusG = 0.0;
static double RadiusP = 0.0;
static double RadiusV = 0.0;
static double Swa = 1.0;
static double Swg = 1.0;
static double Swp = 1.0;
static double Swv = 1.0;
static double N0a = 1.0;
static double N0p = 1.0;
static double R2g = 1.0;

#pragma acc declare create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


#pragma acc routine seq
static double wa(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#else
    return 1.0/Swa * 1.0/(h*h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#endif
}

#pragma acc routine seq
static double dwadr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#else
    return 1.0/Swa * 1.0/(h*h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#endif
}

#pragma acc routine seq
static double wg(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwgdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wp(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swp * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwpdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wv(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swv * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwvdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


	clock_t cFrom, cTill, cStart, cEnd;
	clock_t cNeigh=0, cExplicit=0, cVirial=0, cOther=0;

int main(int argc, char *argv[])
{
	
    char logfilename[1024]  = DEFAULT_LOG;
    char datafilename[1024] = DEFAULT_DATA;
    char gridfilename[1024] = DEFAULT_GRID;
    char proffilename[1024] = DEFAULT_PROF;
    char vtkfilename[1024] = DEFAULT_VTK;
	    int numberofthread = 1;
    
    {
        if(argc>1)strcpy(datafilename,argv[1]);
        if(argc>2)strcpy(gridfilename,argv[2]);
        if(argc>3)strcpy(proffilename,argv[3]);
        if(argc>4)strcpy(vtkfilename,argv[4]);
        if(argc>5)strcpy( logfilename,argv[5]);
    	if(argc>6)numberofthread=atoi(argv[6]);
    }
    log_open(logfilename);
    {
        time_t t=time(NULL);
        log_printf("start reading files at %s\n",ctime(&t));
    }
	{
		#ifdef _OPENMP
		omp_set_num_threads( numberofthread );
		#pragma omp parallel
		{
			if(omp_get_thread_num()==0){
				log_printf("omp_get_num_threads()=%d\n", omp_get_num_threads() );
			}
		}
		#endif
    }
    readDataFile(datafilename);
    readGridFile(gridfilename);
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    initializeWeight();
    initializeFluid();
    initializeWall();
    initializeDomain();

//	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
//	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM],CellParticleBegin[0:TotalCellCount],CellParticleEnd[0:TotalCellCount])
//	#pragma acc enter data create(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
//	#pragma acc enter data create(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
//	#pragma acc enter data create(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
//	#pragma acc enter data create(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
//	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	
	// data transfer from host to device
	#pragma acc update device(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCount,ParticleIndex,Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc update device(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc update device(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc update device(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc update device(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM])
	#pragma acc update device(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount])
	#pragma acc update device(CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	

	{
		calculateCellParticle();
   		calculateNeighbor();
		log_printf("line:%d, NeighborIndCount = %u\n",__LINE__,NeighborIndCount);

		calculateDensityA();
		calculateGravityCenter();
		calculateDensityP();
		writeVtkFile("output.vtk");
		
		{
			time_t t=time(NULL);
			log_printf("start main roop at %s\n",ctime(&t));
		}
		int iStep=(int)(Time/Dt);
		cStart = clock();
		cFrom = cStart;
		while(Time < EndTime + 1.0e-5*Dt){
			
			if( Time + 1.0e-5*Dt >= OutputNext ){
				char filename[256];
				sprintf(filename,proffilename,iStep);
				writeProfFile(filename);
				log_printf("@ Prof Output Time : %e\n", Time );
				OutputNext += OutputInterval;
			}
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;
		
			// particle movement
			calculateConvection();
		
			// wall calculation
			calculateWall();
		
			// periodic boundary calculation
			calculatePeriodicBoundary();
		
			// reset Force to calculate conservative interaction
			resetForce();
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
		
			// calculate Neighbor
			if(neighborCalculation()==1){
				freeNeighbor();
				calculateCellParticle();
				calculateNeighbor();
			}
			cTill = clock(); cNeigh += (cTill-cFrom); cFrom = cTill;
			
			// calculate density
			calculateDensityA();
			calculateGravityCenter();
			calculateDensityP();
			calculateDivergenceP();
			
			// calculate physical coefficient (viscosity, bulk modulus, bulk viscosity..)
			calculatePhysicalCoefficients();
			
			// calculate pressure 
			calculatePressureP();
				
			// calculate P(s,rho) s:fixed
			calculatePressureA();
				
			// calculate diffuse interface force
			calculateDiffuseInterface();
				
			// calculate shear viscosity
			calculateViscosityV();
				
			// calculate Gravity
			calculateGravity();
				
			// calculate intermidiate Velocity
			calculateAcceleration();
				
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			
			if( Time + 1.0e-5*Dt >= VtkOutputNext ){
				calculateVirialStressAtParticle();
				cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;
				
				char filename [256];
				sprintf(filename,vtkfilename,iStep);
				writeVtkFile(filename);
				log_printf("@ Vtk Output Time : %e\n", Time );
				VtkOutputNext += VtkOutputInterval;
				cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

			}
			
			Time += Dt;
			iStep++;
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
		}
	}
	cEnd = cTill;
	
	{
		time_t t=time(NULL);
		log_printf("end main roop at %s\n",ctime(&t));
		log_printf("neighbor search:         %lf [CPU sec]\n", (double)cNeigh/CLOCKS_PER_SEC);
		log_printf("explicit calculation:    %lf [CPU sec]\n", (double)cExplicit/CLOCKS_PER_SEC);
		log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
		log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
		log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cVirial+cOther)/CLOCKS_PER_SEC);
		log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
	}
	
	
	#pragma acc exit data delete(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc exit data delete(Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc exit data delete(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc exit data delete(Mu[0:ParticleCount],Lambda[0:ParticleCount],Kappa[0:ParticleCount],Gravity[0:DIM])
	#pragma acc exit data delete(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
//	#pragma acc exit data delete(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc exit data delete(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc exit data delete(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM])
	#pragma acc exit data delete(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount])
	#pragma acc exit data delete(CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc exit data delete(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc exit data delete(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount],NeighborCalculatedPosition[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc exit data delete(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc exit data delete(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	
	return 0;
	
}

static void readDataFile(char *filename)
{
    FILE * fp;
    char buf[1024];
    const int reading_global=0;
    int mode=reading_global;
    

    fp=fopen(filename,"r");
    mode=reading_global;
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)!=NULL){
            if(buf[0]=='#'){}
            else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
            else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
        	// else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
            else if(sscanf(buf," Density %lf %lf %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3])==4){mode=reading_global;}
            else if(sscanf(buf," BulkModulus %lf %lf %lf %lf",&BulkModulus[0],&BulkModulus[1],&BulkModulus[2],&BulkModulus[3])==4){mode=reading_global;}
            else if(sscanf(buf," BulkViscosity %lf %lf %lf %lf",&BulkViscosity[0],&BulkViscosity[1],&BulkViscosity[2],&BulkViscosity[3])==4){mode=reading_global;}
            else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3])==4){mode=reading_global;}
            else if(sscanf(buf," SurfaceTension %lf %lf %lf %lf",&SurfaceTension[0],&SurfaceTension[1],&SurfaceTension[2],&SurfaceTension[3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type0) %lf %lf %lf %lf",&InteractionRatio[0][0],&InteractionRatio[0][1],&InteractionRatio[0][2],&InteractionRatio[0][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type1) %lf %lf %lf %lf",&InteractionRatio[1][0],&InteractionRatio[1][1],&InteractionRatio[1][2],&InteractionRatio[1][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type2) %lf %lf %lf %lf",&InteractionRatio[2][0],&InteractionRatio[2][1],&InteractionRatio[2][2],&InteractionRatio[2][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type3) %lf %lf %lf %lf",&InteractionRatio[3][0],&InteractionRatio[3][1],&InteractionRatio[3][2],&InteractionRatio[3][3])==4){mode=reading_global;}
            else if(sscanf(buf," Gravity %lf %lf %lf", &Gravity[0], &Gravity[1], &Gravity[2])==3){mode=reading_global;}
            else if(sscanf(buf," Wall2  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[2][0],  &WallCenter[2][1],  &WallCenter[2][2],  &WallVelocity[2][0],  &WallVelocity[2][1],  &WallVelocity[2][2],  &WallOmega[2][0],  &WallOmega[2][1],  &WallOmega[2][2])==9){mode=reading_global;}
            else if(sscanf(buf," Wall3  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[3][0],  &WallCenter[3][1],  &WallCenter[3][2],  &WallVelocity[3][0],  &WallVelocity[3][1],  &WallVelocity[3][2],  &WallOmega[3][0],  &WallOmega[3][1],  &WallOmega[3][2])==9){mode=reading_global;}
            else{
                log_printf("Invalid line in data file \"%s\"\n", buf);
            }
        }
    }
    fclose(fp);
	
	#pragma acc enter data create(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc enter data create(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc enter data create(Gravity[0:DIM])
	#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM])
	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM])
	#pragma acc enter data create(WallRotation[0:WALL_END][0:DIM][0:DIM])
	
}

static void readGridFile(char *filename)
{
    FILE *fp=fopen(filename,"r");
	char buf[1024];   
	
	
	try{
		
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%lf",&Time);
		if(fgets(buf,sizeof(buf),fp)==NULL)throw;
		sscanf(buf,"%d  %lf  %lf %lf %lf  %lf %lf %lf",
			&ParticleCount,
			&ParticleSpacing,
			&DomainMin[0], &DomainMax[0],
			&DomainMin[1], &DomainMax[1],
			&DomainMin[2], &DomainMax[2]);
		#ifdef TWO_DIMENSIONAL
		ParticleVolume = ParticleSpacing*ParticleSpacing;
		#else
		ParticleVolume = ParticleSpacing*ParticleSpacing*ParticleSpacing;
		#endif
		
		ParticleIndex = (int *)malloc(ParticleCount*sizeof(int));
		Property = (int *)malloc(ParticleCount*sizeof(int));
		Position = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Velocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		DensityA = (double *)malloc(ParticleCount*sizeof(double));
		GravityCenter = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		PressureA = (double *)malloc(ParticleCount*sizeof(double));
		VolStrainP = (double *)malloc(ParticleCount*sizeof(double));
		DivergenceP = (double *)malloc(ParticleCount*sizeof(double));
		PressureP = (double *)malloc(ParticleCount*sizeof(double));
		VirialPressureAtParticle = (double *)malloc(ParticleCount*sizeof(double));
		VirialStressAtParticle = (double (*) [DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		Mass = (double (*))malloc(ParticleCount*sizeof(double));
		Force = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Mu = (double (*))malloc(ParticleCount*sizeof(double));
		Lambda = (double (*))malloc(ParticleCount*sizeof(double));
		Kappa = (double (*))malloc(ParticleCount*sizeof(double));
		
		#pragma acc enter data create(ParticleIndex[0:ParticleCount])          attach(ParticleIndex)
		#pragma acc enter data create(Property[0:ParticleCount])               attach(Property)
		#pragma acc enter data create(Position[0:ParticleCount][0:DIM])        attach(Position)
		#pragma acc enter data create(Velocity[0:ParticleCount][0:DIM])        attach(Velocity)
		#pragma acc enter data create(DensityA[0:ParticleCount])               attach(DensityA)
		#pragma acc enter data create(GravityCenter[0:ParticleCount][0:DIM])   attach(GravityCenter)
		#pragma acc enter data create(PressureA[0:ParticleCount])              attach(PressureA)
		#pragma acc enter data create(VolStrainP[0:ParticleCount])             attach(VolStrainP)
		#pragma acc enter data create(DivergenceP[0:ParticleCount])            attach(DivergenceP)
		#pragma acc enter data create(PressureP[0:ParticleCount])              attach(PressureP)
		#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount])               attach(VirialPressureAtParticle)
		#pragma acc enter data create(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])   attach(VirialStressAtParticle)
		#pragma acc enter data create(Mass[0:ParticleCount])                attach(Mass)
		#pragma acc enter data create(Force[0:ParticleCount][0:DIM])        attach(Force)
		#pragma acc enter data create(Mu[0:ParticleCount])                  attach(Mu)
		#pragma acc enter data create(Lambda[0:ParticleCount])              attach(Lambda)
		#pragma acc enter data create(Kappa[0:ParticleCount])               attach(Kappa)
		
		TmpIntScalar = (int *)malloc(ParticleCount*sizeof(int));
		TmpDoubleScalar = (double *)malloc(ParticleCount*sizeof(double));
		TmpDoubleVector = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		NeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
		NeighborCalculatedPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		
		#pragma acc enter data create(TmpIntScalar[0:ParticleCount]) attach(TmpIntScalar)
		#pragma acc enter data create(TmpDoubleScalar[0:ParticleCount]) attach(TmpDoubleScalar)
		#pragma acc enter data create(TmpDoubleVector[0:ParticleCount][0:DIM]) attach(TmpDoubleVector)
		#pragma acc enter data create(NeighborCount[0:ParticleCount]) attach(NeighborCount)
		#pragma acc enter data create(NeighborCalculatedPosition[0:ParticleCount][0:DIM]) attach(NeighborCalculatedPosition)
		
		// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
		ParticleCountPower=0;
		while((ParticleCount>>ParticleCountPower)!=0){
			++ParticleCountPower;
		}
		PowerParticleCount = (1<<ParticleCountPower);
		fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
		CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
		CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
		#pragma acc enter data create(CellIndex   [0:PowerParticleCount]) attach(CellIndex)
		#pragma acc enter data create(CellParticle[0:PowerParticleCount]) attach(CellParticle)

		NeighborPtr  = (long int *)malloc( (PowerParticleCount) * sizeof(long int) );
		#pragma acc enter data create(NeighborPtr[0:PowerParticleCount]) 
		#pragma acc update device(ParticleCountPower,PowerParticleCount)

		
		double (*q)[DIM] = Position;
		double (*v)[DIM] = Velocity;
		
		for(int iP=0;iP<ParticleCount;++iP){
			if(fgets(buf,sizeof(buf),fp)==NULL)break;
			sscanf(buf,"%d  %lf %lf %lf  %lf %lf %lf",
				&Property[iP],
				&q[iP][0],&q[iP][1],&q[iP][2],
				&v[iP][0],&v[iP][1],&v[iP][2]
			);
		}
	}catch(...){};
	
	fclose(fp);
	
	// set particle index
	for(int iP=0;iP<ParticleCount;++iP){
		ParticleIndex[iP]=iP;
	}
	
	// set begin & end
	FluidParticleBegin=0;FluidParticleEnd=0;WallParticleBegin=0;WallParticleEnd=0;
	for(int iP=0;iP<ParticleCount;++iP){
		if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && WALL_BEGIN<=Property[iP+1] && Property[iP+1]<WALL_END){
			FluidParticleEnd=iP+1;
			WallParticleBegin=iP+1;
		}
		if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && iP+1==ParticleCount){
			FluidParticleEnd=iP+1;
		}
		if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END && iP+1==ParticleCount){
			WallParticleEnd=iP+1;
		}
	}
	
	#pragma acc update device(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(Property[0:ParticleCount][0:DIM])
	#pragma acc update device(Position[0:ParticleCount][0:DIM])
	#pragma acc update device(Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	
}

static void writeProfFile(char *filename)
{
    FILE *fp=fopen(filename,"w");

    fprintf(fp,"%e\n",Time);
    fprintf(fp,"%d %e %e %e %e %e %e %e\n",
            ParticleCount,
            ParticleSpacing,
            DomainMin[0], DomainMax[0],
            DomainMin[1], DomainMax[1],
            DomainMin[2], DomainMax[2]);

    const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    for(int iP=0;iP<ParticleCount;++iP){
            fprintf(fp,"%d %e %e %e  %e %e %e\n",
                    Property[iP],
                    q[iP][0], q[iP][1], q[iP][2],
                    v[iP][0], v[iP][1], v[iP][2]
            );
    }
    fflush(fp);
    fclose(fp);
}

static void writeVtkFile(char *filename)
{
	// update parameters to be output
	#pragma acc update host(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount])
	#pragma acc update host(DensityA[0:ParticleCount],VolStrainP[0:ParticleCount])
	#pragma acc update host(NeighborCount[0:ParticleCount],Force[0:ParticleCount][0:DIM])

    const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    FILE *fp=fopen(filename, "w");

    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Unstructured Grid Example\n");
    fprintf(fp, "ASCII\n");

    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)q[iP][0], (float)q[iP][1], (float)q[iP][2]);
    }
    fprintf(fp, "CELLS %d %d\n", ParticleCount, 2*ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 %d ",iP);
    }
    fprintf(fp, "\n");
    fprintf(fp, "CELL_TYPES %d\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 ");
    }
    fprintf(fp, "\n");

    fprintf(fp, "\n");

    fprintf(fp, "POINT_DATA %d\n", ParticleCount);
    fprintf(fp, "SCALARS label float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", Property[iP]);
    }
    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS Mass float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n",(float) Mass[iP]);
//    }
//    fprintf(fp, "\n");
    fprintf(fp, "SCALARS DensityA float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)DensityA[iP]);
    }
    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS PressureA float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)PressureA[iP]);
//    }
//    fprintf(fp, "\n");
    fprintf(fp, "SCALARS VolStrainP float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)VolStrainP[iP]);
    }
    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS DivergenceP float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)DivergenceP[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS PressureP float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)PressureP[iP]);
//    }
//    fprintf(fp, "\n");
    fprintf(fp, "SCALARS VirialPressureAtParticle float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)VirialPressureAtParticle[iP]); // trivial operation is done for 
    }
	fprintf(fp, "\n");
//	for(int iD=0;iD<DIM-1;++iD){
//		for(int jD=0;jD<DIM-1;++jD){
//			fprintf(fp, "SCALARS VirialStressAtParticle[%d][%d] float 1\n",iD,jD);
//			fprintf(fp, "LOOKUP_TABLE default\n");
//			for(int iP=0;iP<ParticleCount;++iP){
//				fprintf(fp, "%e\n", (float)VirialStressAtParticle[iP][iD][jD]); // trivial operation is done for 
//			}
//			fprintf(fp, "\n");    
//		}
//	}
//  
//    fprintf(fp, "SCALARS Mu float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)Mu[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS Lambda float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)Lambda[iP]);
//    }
//    fprintf(fp, "\n");
//    fprintf(fp, "SCALARS Kappa float 1\n");
//    fprintf(fp, "LOOKUP_TABLE default\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e\n", (float)Kappa[iP]);
//    }
//    fprintf(fp, "\n");
    fprintf(fp, "SCALARS neighbor float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", NeighborCount[iP]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "VECTORS velocity float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
    }
    fprintf(fp, "\n");
//    fprintf(fp, "VECTORS GravityCenter float\n");
//    for(int iP=0;iP<ParticleCount;++iP){
//         fprintf(fp, "%e %e %e\n", (float)GravityCenter[iP][0], (float)GravityCenter[iP][1], (float)GravityCenter[iP][2]);
//    }
//    fprintf(fp, "\n");
    fprintf(fp, "VECTORS force float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
    }
    fprintf(fp, "\n");
	
    fflush(fp);
    fclose(fp);
}

static void initializeWeight()
{
	RadiusRatioG = RadiusRatioA;
	
	RadiusA = RadiusRatioA*ParticleSpacing;
	RadiusG = RadiusRatioG*ParticleSpacing;
	RadiusP = RadiusRatioP*ParticleSpacing;
	RadiusV = RadiusRatioV*ParticleSpacing;
	
	
#ifdef TWO_DIMENSIONAL
		Swa = 1.0/2.0 * 2.0/15.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swg = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swp = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swv = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		R2g = 1.0/2.0 * 1.0/30.0* M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing /Swg;
#else	//code for three dimensional
		Swa = 1.0/3.0 * 1.0/5.0*M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swg = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swp = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swv = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		R2g = 1.0/3.0 * 4.0/105.0*M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing/ParticleSpacing /Swg;
#endif
	
	
	    {// N0a
        const double radius_ratio = RadiusA/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusA*RadiusA){
                        const double rij = sqrt(rij2);
                        const double wij = wa(rij,RadiusA);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusA*RadiusA){
                            const double rij = sqrt(rij2);
                            const double wij = wa(rij,RadiusA);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0a = sum;
        log_printf("N0a = %e, count=%d\n", N0a, count);
    }	

    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusP*RadiusP){
                        const double rij = sqrt(rij2);
                        const double wij = wp(rij,RadiusP);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusP*RadiusP){
                            const double rij = sqrt(rij2);
                            const double wij = wp(rij,RadiusP);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }
	
	#pragma acc update device(RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	

}


static void initializeFluid()
{
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Kappa[iP]=BulkModulus[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Lambda[iP]=BulkViscosity[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
	#ifdef TWO_DIMENSIONAL
	CofK = 0.350778153;
	double integN=0.024679383;
	double integX=0.226126699;
	#else 
	CofK = 0.326976006;
	double integN=0.021425779;
	double integX=0.233977488;
	#endif
	
	for(int iT=0;iT<TYPE_COUNT;++iT){
		CofA[iT]=SurfaceTension[iT] / ((RadiusG/ParticleSpacing)*(integN+CofK*CofK*integX));
	}
	
	#pragma acc update device(Mass[0:ParticleCount])
	#pragma acc update device(Kappa[0:ParticleCount])
	#pragma acc update device(Lambda[0:ParticleCount])
	#pragma acc update device(Mu[0:ParticleCount])
	#pragma acc update device(CofK,CofA[0:TYPE_COUNT])
}



static void initializeWall()
{
	
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		
		double theta;
		double normal[DIM]={0.0,0.0,0.0};
		double q[DIM+1];
		double t[DIM];
		double (&R)[DIM][DIM]=WallRotation[iProp];
		
		theta = abs(WallOmega[iProp][0]*WallOmega[iProp][0]+WallOmega[iProp][1]*WallOmega[iProp][1]+WallOmega[iProp][2]*WallOmega[iProp][2]);
		if(theta!=0.0){
			for(int iD=0;iD<DIM;++iD){
				normal[iD]=WallOmega[iProp][iD]/theta;
			}
		}
		q[0]=normal[0]*sin(theta*Dt/2.0);
		q[1]=normal[1]*sin(theta*Dt/2.0);
		q[2]=normal[2]*sin(theta*Dt/2.0);
		q[3]=cos(theta*Dt/2.0);
		t[0]=WallVelocity[iProp][0]*Dt;
		t[1]=WallVelocity[iProp][1]*Dt;
		t[2]=WallVelocity[iProp][2]*Dt;
		
		R[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[0][1] = 2.0*(q[0]*q[1]-q[2]*q[3]);
		R[0][2] = 2.0*(q[0]*q[2]+q[1]*q[3]);
		
		R[1][0] = 2.0*(q[0]*q[1]+q[2]*q[3]);
		R[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[1][2] = 2.0*(q[1]*q[2]-q[0]*q[3]);
		
		R[2][0] = 2.0*(q[0]*q[2]-q[1]*q[3]);
		R[2][1] = 2.0*(q[1]*q[2]+q[0]*q[3]);
		R[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
		
	}
	#pragma acc update device(WallRotation[0:WALL_END][0:DIM][0:DIM])
}

static void initializeDomain( void )
{
	
	
	MaxRadius = ((RadiusA>MaxRadius) ? RadiusA : MaxRadius);
	MaxRadius = ((RadiusG>MaxRadius) ? RadiusG : MaxRadius);
	MaxRadius = ((RadiusP>MaxRadius) ? RadiusP : MaxRadius);
	MaxRadius = ((RadiusV>MaxRadius) ? RadiusV : MaxRadius);

	DomainWidth[0] = DomainMax[0] - DomainMin[0];
	DomainWidth[1] = DomainMax[1] - DomainMin[1];
	DomainWidth[2] = DomainMax[2] - DomainMin[2];
	
	double cellCount[DIM];
	
	cellCount[0] = floor((DomainMax[0] - DomainMin[0])/(MaxRadius+MARGIN));
	cellCount[1] = floor((DomainMax[1] - DomainMin[1])/(MaxRadius+MARGIN));
	#ifdef TWO_DIMENSIONAL
	cellCount[2] = 1;
	#else
	cellCount[2] = floor((DomainMax[2] - DomainMin[2])/(MaxRadius+MARGIN));
	#endif
	
	CellCount[0] = (int)cellCount[0];
	CellCount[1] = (int)cellCount[1];
	CellCount[2] = (int)cellCount[2];
	TotalCellCount   = cellCount[0]*cellCount[1]*cellCount[2];
	log_printf("line:%d CellCount[DIM]: %d %d %d\n", __LINE__, CellCount[0], CellCount[1], CellCount[2]);
	log_printf("line:%d TotalCellCount: %d\n", __LINE__, TotalCellCount);
	
	CellWidth[0]=DomainWidth[0]/CellCount[0];
	CellWidth[1]=DomainWidth[1]/CellCount[1];
	CellWidth[2]=DomainWidth[2]/CellCount[2];
	log_printf("line:%d CellWidth[DIM]: %e %e %e\n", __LINE__, CellWidth[0], CellWidth[1], CellWidth[2]);

	
	CellFluidParticleBegin = (int *)malloc( (TotalCellCount) * sizeof(int) );
	CellFluidParticleEnd   = (int *)malloc( (TotalCellCount) * sizeof(int) );
	CellWallParticleBegin = (int *)malloc( (TotalCellCount) * sizeof(int) );
	CellWallParticleEnd   = (int *)malloc( (TotalCellCount) * sizeof(int) );
	#pragma acc enter data create(CellFluidParticleBegin[0:TotalCellCount]) attach(CellFluidParticleBegin)
	#pragma acc enter data create(CellFluidParticleEnd[0:TotalCellCount]) attach(CellFluidParticleEnd)
	#pragma acc enter data create(CellWallParticleBegin[0:TotalCellCount]) attach(CellWallParticleBegin)
	#pragma acc enter data create(CellWallParticleEnd[0:TotalCellCount]) attach(CellWallParticleEnd)
	
	#pragma acc update device(MaxRadius)	
	#pragma acc update device(CellWidth[0:DIM],CellCount[0:DIM],TotalCellCount)
	#pragma acc update device(DomainMax[0:DIM],DomainMin[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCountPower,PowerParticleCount)
}

static int neighborCalculation( void ){
	double maxShift2=0.0;
	#pragma acc parallel loop reduction (max:maxShift2)
	#pragma omp parallel for reduction (max:maxShift2)
	for(int iP=0;iP<ParticleCount;++iP){
		 double disp[DIM];
         #pragma acc loop seq
         for(int iD=0;iD<DIM;++iD){
            disp[iD] = Mod(Position[iP][iD] - NeighborCalculatedPosition[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
         }
		const double shift2 = disp[0]*disp[0]+disp[1]*disp[1]+disp[2]*disp[2];
		if(shift2>maxShift2){
			maxShift2=shift2;
		}
	}
	
	if(maxShift2>0.5*MARGIN*0.5*MARGIN){
		return 1;
	}
	else{
		return 0;
	}
}


static void calculateCellParticle()
{
	// store and sort with cells
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0; iP<PowerParticleCount; ++iP){
		if(iP<ParticleCount){
			const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth[0]))%CellCount[0];
			const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth[1]))%CellCount[1];
			const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth[2]))%CellCount[2];
			CellIndex[iP]=CellId(iCX,iCY,iCZ);
			if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END){
				CellIndex[iP] += TotalCellCount;
			}
			CellParticle[iP]=iP;
		}
		else{
			CellIndex[ iP ]    = 2*TotalCellCount;
			CellParticle[ iP ] = ParticleCount;
		}
	}
	
	// sort with CellIndex
	// https://edom18.hateblo.jp/entry/2020/09/21/150416
	for(int iMain=0;iMain<ParticleCountPower;++iMain){
		for(int iSub=0;iSub<=iMain;++iSub){
			
			int dist = (1<< (iMain-iSub));
			
			#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
			#pragma acc loop independent
			#pragma omp parallel for
			for(int iP=0;iP<(1<<ParticleCountPower);++iP){
				bool up = ((iP >> iMain) & 2) == 0;
				
				if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
					int tmpCellIndex    = CellIndex[ iP ];
					int tmpCellParticle = CellParticle[ iP ];
					CellIndex[ iP ]     = CellIndex[ iP | dist ];
					CellParticle[ iP ]  = CellParticle[ iP | dist ];
					CellIndex[ iP | dist ]    = tmpCellIndex;
					CellParticle[ iP | dist ] = tmpCellParticle;
				}
			}
		}
	}
	
	// search for CellFluidParticleBegin[iC]
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		CellFluidParticleBegin[iC]=0;
		CellFluidParticleEnd[iC]=0;
		CellWallParticleBegin[iC]=0;
		CellWallParticleEnd[iC]=0;
	}
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=1; iP<ParticleCount+1; ++iP){
		if( CellIndex[iP-1]<CellIndex[iP] ){
			if( CellIndex[iP-1] < TotalCellCount ){
				CellFluidParticleEnd[ CellIndex[iP-1] ] = iP;
			}
			else if( CellIndex[iP-1] - TotalCellCount < TotalCellCount ){
				CellWallParticleEnd[ CellIndex[iP-1]-TotalCellCount ] = iP;
			}
			if( CellIndex[iP] < TotalCellCount ){
				CellFluidParticleBegin[ CellIndex[iP] ] = iP;
			}
			else if( CellIndex[iP] - TotalCellCount < TotalCellCount ){
				CellWallParticleBegin[ CellIndex[iP]-TotalCellCount ] = iP;
			}
		}
	}
	
	// Fill zeros in CellParticleBegin and CellParticleEnd
	int power = 0;
	const int N = 2*TotalCellCount;
	while( (N>>power) != 0 ){
		power++;
	}
	const int powerN = (1<<power);
	
	int * ptr = (int *)malloc( powerN * sizeof(int));
	#pragma acc enter data create(ptr[0:powerN])
	
	#pragma acc kernels present(ptr[0:powerN])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<powerN;++iRow){
		ptr[iRow]=0;
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		ptr[iC]               =CellFluidParticleEnd[iC]-CellFluidParticleBegin[iC];
		ptr[iC+TotalCellCount]=CellWallParticleEnd[iC] -CellWallParticleBegin[iC];
	}
	
	// Convert ptr to cumulative sum
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (1<<iMain);	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]+=ptr[iRow+dist];
		}
	}
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (powerN>>(iMain+1));	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]-=ptr[iRow+dist];
			ptr[iRow+dist]+=ptr[iRow];
		}
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		if(iC==0){	CellFluidParticleBegin[iC]=0;	}
		else     { 	CellFluidParticleBegin[iC]=ptr[iC-1];	}
		CellFluidParticleEnd[iC]  =ptr[iC];
		CellWallParticleBegin[iC] =ptr[iC-1+TotalCellCount];
		CellWallParticleEnd[iC]   =ptr[iC+TotalCellCount];
	}
	
	free(ptr);
	#pragma acc exit data delete(ptr[0:powerN])
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	{
		FluidParticleBegin = CellFluidParticleBegin[0];
		FluidParticleEnd   = CellFluidParticleEnd[TotalCellCount-1];
		WallParticleBegin  = CellWallParticleBegin[0];
		WallParticleEnd    = CellWallParticleEnd[TotalCellCount-1];
	}
	#pragma acc update host(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
//	fprintf(stderr,"line:%d, FluidParticleBegin=%d, FluidParticleEnd=%d, WallParticleBegin=%d, WallParticleEnd=%d\n",__LINE__,FluidParticleBegin, FluidParticleEnd, WallParticleBegin, WallParticleEnd);
	
	// re-arange particles in CellIndex order
	#pragma acc kernels present(ParticleIndex[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=ParticleIndex[CellParticle[iP]];
	}
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		ParticleIndex[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=Property[CellParticle[iP]];
	}
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Property[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Position[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Position[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}
		
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Velocity[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Velocity[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}
}


static void calculateNeighbor( void )
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		NeighborCount[iP]=0;
	}
	
	const int rangeX = (int)(ceil((MaxRadius+MARGIN)/CellWidth[0]));
	const int rangeY = (int)(ceil((MaxRadius+MARGIN)/CellWidth[1]));
	#ifdef TWO_DIMENSIONAL
	const int rangeZ = 0;
	#else // not TWO_DIMENSIONAL (three dimensional) 
	const int rangeZ = (int)(ceil((MaxRadius+MARGIN)/CellWidth[2]));
	#endif
	
	#define MAX_1D_NEIGHBOR_CELL_COUNT 3
	assert( 2*rangeX+1 <= MAX_1D_NEIGHBOR_CELL_COUNT );
	assert( 2*rangeY+1 <= MAX_1D_NEIGHBOR_CELL_COUNT );
	assert( 2*rangeZ+1 <= MAX_1D_NEIGHBOR_CELL_COUNT );
	
	
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		const int iCX=(CellIndex[iP]/(CellCount[1]*CellCount[2]))%TotalCellCount;
		const int iCY=(CellIndex[iP]/CellCount[2])%CellCount[1];
		const int iCZ=CellIndex[iP]%CellCount[2];
		
		int jCXs[MAX_1D_NEIGHBOR_CELL_COUNT];
		int jCYs[MAX_1D_NEIGHBOR_CELL_COUNT];
		int jCZs[MAX_1D_NEIGHBOR_CELL_COUNT];
		
		#pragma acc loop seq
		for(int jX=0;jX<2*rangeX+1;++jX){
			jCXs[jX]=((iCX-rangeX+jX)%CellCount[0]+CellCount[0])%CellCount[0];
		}
		#pragma acc loop seq
		for(int jY=0;jY<2*rangeY+1;++jY){
			jCYs[jY]=((iCY-rangeY+jY)%CellCount[1]+CellCount[1])%CellCount[1];
		}
		#pragma acc loop seq
		for(int jZ=0;jZ<2*rangeZ+1;++jZ){
			jCZs[jZ]=((iCZ-rangeZ+jZ)%CellCount[2]+CellCount[2])%CellCount[2];
		}
		const int bX = (2*rangeX)-(iCX+rangeX)%CellCount[0];
		const int jXmin= ( ( bX>0 )? bX:0 );
		const int bY = (2*rangeY)-(iCY+rangeY)%CellCount[1];
		const int jYmin= ( ( bY>0 )? bY:0 );
		const int bZ = (2*rangeZ)-(iCZ+rangeZ)%CellCount[2];
		const int jZmin= ( ( bZ>0 )? bZ:0 );
		
		#pragma acc loop seq
		for(int jX=jXmin;jX<jXmin+(2*rangeX+1);++jX){
			#pragma acc loop seq
			for(int jY=jYmin;jY<jYmin+(2*rangeY+1);++jY){
				#pragma acc loop seq
				for(int jZ=jZmin;jZ<jZmin+(2*rangeZ+1);++jZ){
					const int jCX = jCXs[ jX % (2*rangeX+1)];
					const int jCY = jCYs[ jY % (2*rangeY+1)];
					const int jCZ = jCZs[ jZ % (2*rangeZ+1)];
					const int jC=CellId(jCX,jCY,jCZ);
					#pragma acc loop seq
					for(int jP=CellFluidParticleBegin[jC];jP<CellFluidParticleEnd[jC];++jP){
						double qij[DIM];
						#pragma acc loop seq
						for(int iD=0;iD<DIM;++iD){
							qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
						}
						const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
						#ifndef _OPENACC
						if( iP!=jP && qij2==0.0 ){
							log_printf("line:%d, Warning:overlaped iP=%d, jP=%d\n", __LINE__, iP, jP);
							log_printf("x[iP] = %e, %e, %e\n", Position[iP][0],Position[iP][1],Position[iP][2]);
							log_printf("v[iP] = %e, %e, %e\n", Velocity[iP][0],Velocity[iP][1],Velocity[iP][2]);
						}
						#endif
						if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
							NeighborCount[iP]++;
						}
					}
				}
			}
		}
		
		if( WallParticleBegin<=iP && iP<WallParticleEnd && NeighborCount[iP]==0)continue;
		
		#pragma acc loop seq
		for(int jX=jXmin;jX<jXmin+(2*rangeX+1);++jX){
			#pragma acc loop seq
			for(int jY=jYmin;jY<jYmin+(2*rangeY+1);++jY){
				#pragma acc loop seq
				for(int jZ=jZmin;jZ<jZmin+(2*rangeZ+1);++jZ){
					const int jCX = jCXs[ jX % (2*rangeX+1)];
					const int jCY = jCYs[ jY % (2*rangeY+1)];
					const int jCZ = jCZs[ jZ % (2*rangeZ+1)];
					const int jC=CellId(jCX,jCY,jCZ);
					#pragma acc loop seq
					for(int jP=CellWallParticleBegin[jC];jP<CellWallParticleEnd[jC];++jP){
						double qij[DIM];
						#pragma acc loop seq
						for(int iD=0;iD<DIM;++iD){
							qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
						}
						const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
						#ifndef _OPENACC
						if( iP!=jP && qij2==0.0 ){
							log_printf("line:%d, Warning:overlaped iP=%d, jP=%d\n", __LINE__, iP, jP);
							log_printf("x[iP] = %e, %e, %e\n", Position[iP][0],Position[iP][1],Position[iP][2]);
							log_printf("v[iP] = %e, %e, %e\n", Velocity[iP][0],Velocity[iP][1],Velocity[iP][2]);
						}
						#endif
						if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
							NeighborCount[iP]++;
						}
					}
				}
			}
		}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<PowerParticleCount;++iP){
		NeighborPtr[iP]=0;
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		NeighborPtr[iP+1]=NeighborCount[iP];
	}
	
	// Convert NeighborPtr & NeighborPtrP into cumulative sum
	for(int iMain=0;iMain<ParticleCountPower;++iMain){
		const int dist = (1<<iMain);	
		#pragma acc kernels present(NeighborPtr[0:PowerParticleCount])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<PowerParticleCount;iP+=(dist<<1)){
			NeighborPtr[iP] += NeighborPtr[iP+dist];
		}
	}
    for(int iMain=0;iMain<ParticleCountPower;++iMain){
		const int dist = (PowerParticleCount>>(iMain+1));	
		#pragma acc kernels present(NeighborPtr[0:PowerParticleCount])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<PowerParticleCount;iP+=(dist<<1)){
			NeighborPtr[iP] -= NeighborPtr[iP+dist];
			NeighborPtr[iP+dist] += NeighborPtr[iP];
		}
	}
	
	#pragma acc kernels present(NeighborPtr[0:PowerParticleCount])
	{
		NeighborIndCount = NeighborPtr[ParticleCount];
	}
	#pragma acc update host(NeighborIndCount)
	//log_printf("line:%d, NeighborIndCount = %u\n",__LINE__,NeighborIndCount);
    
	NeighborInd = (int *)malloc( NeighborIndCount * sizeof(int) );
	#pragma acc enter data create(NeighborInd[0:NeighborIndCount])
	
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		const int iCX=(CellIndex[iP]/(CellCount[1]*CellCount[2]))%TotalCellCount;
		const int iCY=(CellIndex[iP]/CellCount[2])%CellCount[1];
		const int iCZ=CellIndex[iP]%CellCount[2];
		
		int jCXs[MAX_1D_NEIGHBOR_CELL_COUNT];
		int jCYs[MAX_1D_NEIGHBOR_CELL_COUNT];
		int jCZs[MAX_1D_NEIGHBOR_CELL_COUNT];
		
		#pragma acc loop seq
		for(int jX=0;jX<2*rangeX+1;++jX){
			jCXs[jX]=((iCX-rangeX+jX)%CellCount[0]+CellCount[0])%CellCount[0];
		}
		#pragma acc loop seq
		for(int jY=0;jY<2*rangeY+1;++jY){
			jCYs[jY]=((iCY-rangeY+jY)%CellCount[1]+CellCount[1])%CellCount[1];
		}
		#pragma acc loop seq
		for(int jZ=0;jZ<2*rangeZ+1;++jZ){
			jCZs[jZ]=((iCZ-rangeZ+jZ)%CellCount[2]+CellCount[2])%CellCount[2];
		}
		const int bX = (2*rangeX)-(iCX+rangeX)%CellCount[0];
		const int jXmin= ( ( bX>0 )? bX:0 );
		const int bY = (2*rangeY)-(iCY+rangeY)%CellCount[1];
		const int jYmin= ( ( bY>0 )? bY:0 );
		const int bZ = (2*rangeZ)-(iCZ+rangeZ)%CellCount[2];
		const int jZmin= ( ( bZ>0 )? bZ:0 );
		
		int iN = 0;
		
		#pragma acc loop seq
		for(int jX=jXmin;jX<jXmin+(2*rangeX+1);++jX){
			#pragma acc loop seq
			for(int jY=jYmin;jY<jYmin+(2*rangeY+1);++jY){
				#pragma acc loop seq
				for(int jZ=jZmin;jZ<jZmin+(2*rangeZ+1);++jZ){
					const int jCX = jCXs[ jX % (2*rangeX+1)];
					const int jCY = jCYs[ jY % (2*rangeY+1)];
					const int jCZ = jCZs[ jZ % (2*rangeZ+1)];
					const int jC=CellId(jCX,jCY,jCZ);
					#pragma acc loop seq
					for(int jP=CellFluidParticleBegin[jC];jP<CellFluidParticleEnd[jC];++jP){
						double qij[DIM];
						#pragma acc loop seq
						for(int iD=0;iD<DIM;++iD){
							qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
						}
						const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
						if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
							NeighborInd[ NeighborPtr[iP]+iN ] = jP;
							iN++;
						}
					}
				}
			}
		}
		
		if( WallParticleBegin<=iP && iP<WallParticleEnd && NeighborCount[iP]==0)continue;
		
		#pragma acc loop seq
		for(int jX=jXmin;jX<jXmin+(2*rangeX+1);++jX){
			#pragma acc loop seq
			for(int jY=jYmin;jY<jYmin+(2*rangeY+1);++jY){
				#pragma acc loop seq
				for(int jZ=jZmin;jZ<jZmin+(2*rangeZ+1);++jZ){
					const int jCX = jCXs[ jX % (2*rangeX+1)];
					const int jCY = jCYs[ jY % (2*rangeY+1)];
					const int jCZ = jCZs[ jZ % (2*rangeZ+1)];
					const int jC=CellId(jCX,jCY,jCZ);
					#pragma acc loop seq
					for(int jP=CellWallParticleBegin[jC];jP<CellWallParticleEnd[jC];++jP){
						double qij[DIM];
						#pragma acc loop seq
						for(int iD=0;iD<DIM;++iD){
							qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
						}
						const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
						if(qij2 <= (MaxRadius+MARGIN)*(MaxRadius+MARGIN)){
							NeighborInd[ NeighborPtr[iP]+iN ] = jP;
							iN++;
						}
					}
				}
			}
		}
	}
	
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			NeighborCalculatedPosition[iP][iD]=Position[iP][iD];
		}
	}
}

static void freeNeighbor()
{
	free(NeighborInd);
	#pragma acc exit data delete(NeighborInd)
}


static void calculateConvection()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Position[iP][0] += Velocity[iP][0]*Dt;
        Position[iP][1] += Velocity[iP][1]*Dt;
        Position[iP][2] += Velocity[iP][2]*Dt;
    }
}

static void resetForce()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]=0.0;
        }
    }
}


static void calculatePhysicalCoefficients()
{	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Kappa[iP]=BulkModulus[Property[iP]];
		if(VolStrainP[iP]<0.0){Kappa[iP]=0.0;}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Lambda[iP]=BulkViscosity[Property[iP]];
//		if(VolStrainP[iP]<0.0){Lambda[iP]=0.0;}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
}


static void calculateDensityA()
{ 
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double ratio = InteractionRatio[Property[iP]][Property[jP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double radius = RadiusA;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = ratio * wa(rij,radius);
				sum += weight;
			}
		}
		DensityA[iP]=sum;
	}
}


static void calculateGravityCenter()
{
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double ratio = InteractionRatio[Property[iP]][Property[jP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = ratio * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					sum[iD] += xij[iD]*weight/R2g*RadiusG;
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			GravityCenter[iP][iD] = sum[iD];
		}
	}
}

static void calculatePressureA()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureA[iP] = CofA[Property[iP]]*(DensityA[iP]-N0a)/ParticleSpacing;
		if(N0a<=DensityA[iP]){
			PressureA[iP] = 0.0;
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],PressureA[0:ParticleCount],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
        	double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusA;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = ratio_ij * dwadr(rij,radius);
            	const double dwji = ratio_ji * dwadr(rij,radius);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
            	#pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureA[iP]*dwij+PressureA[jP]*dwji)*eij[iD]* ParticleVolume;
                }
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}

static void calculateDiffuseInterface()
{
	
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],GravityCenter[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		const double ai = CofA[Property[iP]]*(CofK)*(CofK);
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			const double aj = CofA[Property[iP]]*(CofK)*(CofK);
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
			double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 > 0){
				const double rij = sqrt(rij2);
				const double wij = ratio_ij * wg(rij,RadiusG);
				const double wji = ratio_ji * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (aj*GravityCenter[jP][iD]*wji-ai*GravityCenter[iP][iD]*wij)/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				const double dwij = ratio_ij * dwgdr(rij,RadiusG);
				const double dwji = ratio_ji * dwgdr(rij,RadiusG);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (aj*GravityCenter[jP][iD]*dwji-ai*GravityCenter[iP][iD]*dwij)*xij[iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (gr)*eij[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateDensityP()
{
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = wp(rij,radius);
				sum += weight;
			}
		}
		VolStrainP[iP] = (sum - N0p);
	}
}

static void calculateDivergenceP()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					sum -= uij[iD]*eij[iD]*dw;
				}
			}
		}
		DivergenceP[iP]=sum;
	}
}

static void calculatePressureP()
{
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureP[iP] = -Lambda[iP]*DivergenceP[iP]+Kappa[iP]*VolStrainP[iP];
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],PressureP[0:ParticleCount],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] += (PressureP[iP]+PressureP[jP])*gradw[iD]*ParticleVolume;
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateViscosityV(){

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Mu[0:ParticleCount],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
        	if(RadiusV*RadiusV - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,RadiusV);
            	const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
        		double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
            	double fij[DIM] = {0.0,0.0,0.0};
        		#pragma acc loop seq
            	for(int iD=0;iD<DIM;++iD){
            		#ifdef TWO_DIMENSIONAL
            		force[iD] += 8.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#else
            		force[iD] += 10.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#endif
            	}
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}


static void calculateGravity(){
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Force[iP][0] += Mass[iP]*Gravity[0];
        Force[iP][1] += Mass[iP]*Gravity[1];
        Force[iP][2] += Mass[iP]*Gravity[2];
    }
}

static void calculateAcceleration()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
        Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
        Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
    }
}


static void calculateWall()
{
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
        Force[iP][0] = 0.0;
        Force[iP][1] = 0.0;
        Force[iP][2] = 0.0;
    }
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
		const int iProp = Property[iP];
		double r[DIM] = {Position[iP][0]-WallCenter[iProp][0],Position[iP][1]-WallCenter[iProp][1],Position[iP][2]-WallCenter[iProp][2]};
		const double (&R)[DIM][DIM] = WallRotation[iProp];
		const double (&w)[DIM] = WallOmega[iProp];
		r[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2];
		r[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2];
		r[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2];
		Velocity[iP][0] = w[1]*r[2]-w[2]*r[1] + WallVelocity[iProp][0];
		Velocity[iP][1] = w[2]*r[0]-w[0]*r[2] + WallVelocity[iProp][1];
		Velocity[iP][2] = w[0]*r[1]-w[1]*r[0] + WallVelocity[iProp][2];
		Position[iP][0] = r[0] + WallCenter[iProp][0] + WallVelocity[iProp][0]*Dt;
		Position[iP][1] = r[1] + WallCenter[iProp][1] + WallVelocity[iProp][1]*Dt;
		Position[iP][2] = r[2] + WallCenter[iProp][2] + WallVelocity[iProp][2]*Dt;
		
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		WallCenter[iProp][0] += WallVelocity[iProp][0]*Dt;
		WallCenter[iProp][1] += WallVelocity[iProp][1]*Dt;
		WallCenter[iProp][2] += WallVelocity[iProp][2]*Dt;
	}
}



static void calculateVirialStressAtParticle()
{
	const double (*x)[DIM] = Position;
	const double (*v)[DIM] = Velocity;
	

	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD]=0.0;
			}
		}
	}
	
	#pragma acc kernels present(x[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			// pressureP
			if(RadiusP*RadiusP - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = dwpdr(rij,RadiusP);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureP[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// pressureA
			if(RadiusA*RadiusA - rij2 > 0){
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dwij = ratio * dwadr(rij,RadiusA);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureA[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}

	}
	
	#pragma acc kernels present(x[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Mu[0:ParticleCount],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// viscosity term
			if(RadiusV*RadiusV - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = -dwvdr(rij,RadiusV);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				const double vij[DIM] = {v[jP][0]-v[iP][0],v[jP][1]-v[iP][1],v[jP][2]-v[iP][2]};
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#ifdef TWO_DIMENSIONAL
					fij[iD] = 8.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#else
					fij[iD] = 10.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#endif
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=0.5*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM],NeighborInd[0:NeighborIndCount])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// diffuse interface force (1st term)
			if(RadiusG*RadiusG - rij2 > 0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double weight = ratio * wg(rij,RadiusG);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*( -GravityCenter[iP][iD])*weight/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
			
			// diffuse interface force (2nd term)
			if(RadiusG*RadiusG - rij2 > 0.0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dw = ratio * dwgdr(rij,RadiusG);
				const double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (                     -GravityCenter[iP][iD])*xij[iD];
				}
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*(gr)*gradw[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}	
	

	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#ifdef TWO_DIMENSIONAL
		VirialPressureAtParticle[iP]=-1.0/2.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]);
		#else 
		VirialPressureAtParticle[iP]=-1.0/3.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]+VirialStressAtParticle[iP][2][2]);
		#endif
	}

}



static void calculatePeriodicBoundary( void )
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Position[iP][iD] = Mod(Position[iP][iD]-DomainMin[iD],DomainWidth[iD])+DomainMin[iD];
        }
    }
}

