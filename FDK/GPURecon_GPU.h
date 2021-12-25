#ifndef __GPURecon_GPU_H__
#define ___GPURecon_GPU_H__
#define PI 3.14159265359f

#define SAFE_DELETE(pPointer) if(pPointer){delete [] pPointer; pPointer=NULL;}

//typedef int* intPtr;
//typedef float* floatPtr;

struct RotInfo
{
	int nCols;
	int nRows;
	float fDetLeft;
	float fDetBottom;
	float fDeltaCol;
	float fDeltaRow;
	float fR;
	float fD;
	float fStartAngle;
	int nViews;
	float fDeltaAngle;
	float fOffSet;
	int nSample; 
};

struct ViewInfo 
{
	float fSourceZ;
	float fAngle;
	float fDeltaZ; // fDeltaZ is not 0 when using Z warbling
	float fDeltaR;
};

struct ImageData
{
	int nX;
	int nY;
	int nZ;
	float fXMin;
	float fYMin;
	float fZMin;
	float fDeltaX;
	float fDeltaY;
	float fDeltaZ;
};

struct ViewData:ViewInfo
{
	float * pfView;
};

//void ApplyParkerWeighting(float * pfProjection, const RotInfo ri);

//void ApplyCBWeighting(float * pfProjection, const float* pfProjection, const RotInfo ri);

//void Filtering(const float * pfProjection, float* pfProjection_, const RotInfo ri); 

void BP_One_Rot_3D_GPU(const struct ImageData & idImage, const float* pfProjection,\
float* pfData,  const float * anglePos, const RotInfo & ri, float fFOVRadius);

//extern "C" void LCFBP_GPU(const struct ImageData & idImage, const struct ViewData* pvdViews, float * pfProj2, const RotInfo & ri, float fFOVRadius);

void SelectCUDADevice(int nDev = -1);

#endif // ___GPURecon_GPU_H__
