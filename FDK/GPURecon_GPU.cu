//#include "CommonDef.h"
#include "GPURecon_GPU.h"
//#include <cutil_inline.h>
#include<helper_cuda.h>
#include "ipp_m7.h"
#include "ipp.h"

texture<float, 2, cudaReadModeElementType> prjTexture2D;
texture<float, 2, cudaReadModeElementType> weightTexture2D;
texture<float, 3, cudaReadModeElementType> imgTexture3D;
texture<float, 2, cudaReadModeElementType> singleViewTex;
texture<float, 3, cudaReadModeElementType> prjTexture3D;


__global__ void
BackProjOneView_CB_EA_kernel(RotInfo ri, ViewData vd, ImageData id, float * pfData, float fSinAngle, float fCosAngle, int nImagePitch, float fFOVRadiusSquare) 
{
    // calculate normalized texture coordinates
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	
	float fX, fY, fZ, fOneOverProjDistSquared, fColIndex, fRowIndex, fU, fDistanceToCenterRay;

	if(i<id.nX && j<id.nY)
	{
		
		fX = id.fXMin+id.fDeltaX*(i+0.5f);
		fY = id.fYMin+id.fDeltaY*(j+0.5f);	
		
		if(fX*fX+fY*fY<=fFOVRadiusSquare)
		{
			fU = ri.fR+fX*fCosAngle+fY*fSinAngle;		//projection of point onto iso-ray
			fDistanceToCenterRay = -fX*fSinAngle+fY*fCosAngle;
			fOneOverProjDistSquared=1/(fU*fU+fDistanceToCenterRay*fDistanceToCenterRay);
			
			fColIndex = (atanf(fDistanceToCenterRay/fU)-ri.fDetLeft)/ri.fDeltaCol + ri.fOffSet; //note the -0.5 is not here as usual because of the texture indexing convention 
			
			float * pfVoxel = pfData+j*id.nX+i;	
			if(fColIndex>0 && fColIndex<ri.nCols)
				for(int k=0; k<id.nZ; k++)
				{
					fZ= id.fZMin+id.fDeltaZ*(k+0.5f) ;
					fRowIndex = ((fZ-vd.fSourceZ)*ri.fD/fU-ri.fDetBottom)/ri.fDeltaRow;
					* pfVoxel += tex2D(singleViewTex, fColIndex,  fRowIndex ) * fOneOverProjDistSquared ;
					pfVoxel += nImagePitch;
				}
		}
	}
}

void Filtering(const float * pfProjection, float * pfProjection_, const RotInfo ri){
	float * pfConvKernel = new float[2*ri.nCols-1];
	memset(pfConvKernel,0,(2*ri.nCols-1)*sizeof(float));
	int nIndexRow;

	//float * pfProjection_ = NULL; 

	pfProjection_ = (float*)malloc(ri.nCols*ri.nViews *sizeof(float));
	//checkCudaErrors(cudaMallocHost((void**) &pfProjection_, ri.nCols*ri.nViews*sizeof(float)));
    //checkCudaErrors(cudaMemset(pfProjection_, 0, ri.nCols*ri.nViews*sizeof(float)));
	checkCudaErrors(cudaMemcpy(pfProjection_, pfProjection, ri.nCols*ri.nViews*sizeof(float), cudaMemcpyHostToHost));


 	//////////////////////////////////////////////////////////////////////////
 	//  Ramp kernel using the h'' form in Hsieh's book, Eq 3.51
 	for(nIndexRow=-ri.nCols+1;nIndexRow<=ri.nCols-1;nIndexRow++)
 	{
 		if(nIndexRow == 0)
 			pfConvKernel[nIndexRow+ri.nCols-1]=0.5f/(4*ri.fDeltaCol*ri.fDeltaCol);
 		else if(nIndexRow%2)
 			pfConvKernel[nIndexRow+ri.nCols-1]= -0.5f/(PI*PI*sinf(nIndexRow*ri.fDeltaCol)*sinf(nIndexRow*ri.fDeltaCol));
 		else
 		    pfConvKernel[nIndexRow+ri.nCols-1]= 0;
 	}	
	
	ippsMulC_32f_I(ri.fR*ri.fDeltaAngle*ri.fDeltaCol,pfConvKernel,ri.nCols*2-1);
	
	float * pfBuffer = new float[3*ri.nCols-2];
	//////////////////////////////////////////////////////////////////////////
	//  convolution
	float * pPrj;
	int nAngle,nRow;
	for(nAngle=0; nAngle<ri.nViews; nAngle++)
	{
		for(nRow=0; nRow<ri.nRows; nRow++)
		{
			pPrj = pfProjection_+nAngle*ri.nCols*ri.nRows+nRow*ri.nCols;
			ippsConv_32f(pPrj,ri.nCols,pfConvKernel,ri.nCols*2-1,pfBuffer);	
			memcpy(pPrj,pfBuffer+ri.nCols-1,ri.nCols*sizeof(float));
		}
	}
	SAFE_DELETE(pfBuffer);
	SAFE_DELETE(pfConvKernel);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

void SelectCUDADevice(int nDev)
{
	//int deviceCount;
    int dev, nSelected;
	cudaDeviceProp deviceProp;
	//size_t nTotal;
	
	nSelected = nDev;
	checkCudaErrors(cudaSetDevice(nSelected));
	checkCudaErrors(cudaGetDevice(&dev));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	//printf("Device %d: \"%s\" is used.\n", dev, deviceProp.name);
}

inline void SetupSinoTexture(cudaArray* cu_sinoOneView)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	prjTexture2D.addressMode[0] = cudaAddressModeClamp;
	prjTexture2D.addressMode[1] = cudaAddressModeClamp;
	prjTexture2D.filterMode = cudaFilterModeLinear;
	prjTexture2D.normalized = false;    // access with normalized texture coordinates
	checkCudaErrors(cudaBindTextureToArray(prjTexture2D, cu_sinoOneView, channelDesc));
}

inline void SetupWeightTexture(cudaArray* cu_weightOneView)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	weightTexture2D.addressMode[0] = cudaAddressModeClamp;
	weightTexture2D.addressMode[1] = cudaAddressModeClamp;
	weightTexture2D.filterMode = cudaFilterModeLinear;
	weightTexture2D.normalized = false;    // access with normalized texture coordinates
	checkCudaErrors(cudaBindTextureToArray(weightTexture2D, cu_weightOneView, channelDesc));
}

inline void SetupImageTexture(cudaArray* cu_imgArray)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    // set texture parameters
    imgTexture3D.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
    imgTexture3D.addressMode[1] = cudaAddressModeClamp;
    imgTexture3D.addressMode[2] = cudaAddressModeClamp;
    imgTexture3D.normalized = false;                      // access with un-normalized texture coordinates
    imgTexture3D.filterMode = cudaFilterModeLinear;      // linear interpolation
	
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(imgTexture3D, cu_imgArray, channelDesc));
}

inline void CopyImageTo3DArray(cudaArray* cu_imgArray, float* d_imgArray, const struct ImageData & iiImage, int nBPZStart, int nBPZ)
{
	// copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_imgArray, iiImage.nX*sizeof(float), iiImage.nX ,iiImage.nY);
    copyParams.dstArray = cu_imgArray;
    copyParams.extent   = make_cudaExtent(iiImage.nX,iiImage.nY,nBPZ);
	copyParams.dstPos = make_cudaPos(0,0,nBPZStart);
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
}

/*===========================================
  New version
===========================================*/

void BP_One_Rot_3D_GPU(const struct ImageData & idImage, const float* pfProjection, \
	float* pfData, const float * anglePos, \
	const RotInfo & ri, float fFOVRadius)
{
	int nViewSize = ri.nCols*sizeof(float);
	int nImagePitch = idImage.nX*idImage.nY; 
	int nImageSize = nImagePitch*sizeof(float);
	float fFOVRadiusSquare = fFOVRadius* fFOVRadius;
	int n;

	float * pfProjection_ = NULL; 

	ViewData * pvdViews;
	pvdViews = new ViewData[ri.nViews];

	pfProjection_ = (float*)malloc(ri.nCols*ri.nViews *sizeof(float));
	checkCudaErrors(cudaMemcpy(pfProjection_, pfProjection, ri.nCols*ri.nViews*sizeof(float), cudaMemcpyHostToHost));

	/*===========================================
    Filtering
    ===========================================*/
	float * pfConvKernel = new float[2*ri.nCols-1]();
	memset(pfConvKernel,0,(2*ri.nCols-1)*sizeof(float));
	int nIndexRow;

 	////////////////////////////////////////////////////////////////////////
 	//  Ramp kernel using the h'' form in Hsieh's book, Eq 3.51
 	for(nIndexRow=-ri.nCols+1;nIndexRow<=ri.nCols-1;nIndexRow++)
 	{
 		if(nIndexRow == 0)
 			pfConvKernel[nIndexRow+ri.nCols-1]=0.5f/(4*ri.fDeltaCol*ri.fDeltaCol);
 		else if(nIndexRow%2)
 			pfConvKernel[nIndexRow+ri.nCols-1]= -0.5f/(PI*PI*sinf(nIndexRow*ri.fDeltaCol)*sinf(nIndexRow*ri.fDeltaCol));
 		else
 		    pfConvKernel[nIndexRow+ri.nCols-1]= 0;
 	}	
	
	ippsMulC_32f_I(ri.fR*ri.fDeltaAngle*ri.fDeltaCol,pfConvKernel,ri.nCols*2-1);
	
	float * pfBuffer = new float[3*ri.nCols-2];
	//////////////////////////////////////////////////////////////////////////
	//  convolution
	float * pPrj;
	int nAngle,nRow;
	for(nAngle=0; nAngle<ri.nViews; nAngle++)
	{
		for(nRow=0; nRow<ri.nRows; nRow++)
		{
			pPrj = pfProjection_+nAngle*ri.nCols*ri.nRows+nRow*ri.nCols;
			ippsConv_32f(pPrj,ri.nCols,pfConvKernel,ri.nCols*2-1,pfBuffer);	
			memcpy(pPrj,pfBuffer+ri.nCols-1,ri.nCols*sizeof(float));
		}
	}
	SAFE_DELETE(pfBuffer);
	SAFE_DELETE(pfConvKernel);

	/*===========================================
    Filtering Ending 
    ===========================================*/

	for(int nTmp=0; nTmp<ri.nViews; nTmp++)
	  {
		  pvdViews[nTmp].fAngle = anglePos[nTmp]; // ri.fStartAngle+nTmp*ri.fDeltaAngle;
		  pvdViews[nTmp].fSourceZ = 0;
		  pvdViews[nTmp].pfView = pfProjection_ + nTmp*ri.nCols; 
	  }

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cu_sinoOneView;
	checkCudaErrors(cudaMallocArray( &cu_sinoOneView, &channelDesc, ri.nCols, 1)); 
	
	
	//allocate device memory for the image matrix
    float* dev_imgArray = NULL;
    checkCudaErrors( cudaMalloc( (void**) &dev_imgArray, nImageSize) );	
    //checkCudaErrors( cudaMemcpy( dev_imgArray , pfData, nImageSize, cudaMemcpyHostToDevice ) );
	ImageData dev_idImage;
	memcpy(&dev_idImage, &idImage, sizeof(idImage));
    dim3 dimBlockImg(8, 8, 1);
    dim3 dimGridImg( (idImage.nX+dimBlockImg.x-1)/dimBlockImg.x, (idImage.nY+dimBlockImg.y-1)/dimBlockImg.y, 1);

	singleViewTex.addressMode[0] = cudaAddressModeClamp;
	singleViewTex.addressMode[1] = cudaAddressModeClamp;
	singleViewTex.filterMode = cudaFilterModeLinear;
	singleViewTex.normalized = false;    // access with normalized texture coordinates
	checkCudaErrors( cudaBindTextureToArray( singleViewTex, cu_sinoOneView, channelDesc));
	for(n=0; n<ri.nViews; n++)
	{
//		cout << "|=>  >> Processing View " << n << "/" << ri.nViews << "          \r" << std::flush;
		// execute the kernel
		checkCudaErrors( cudaMemcpyToArray( cu_sinoOneView, 0, 0, pvdViews[n].pfView, nViewSize, cudaMemcpyHostToDevice));	
		BackProjOneView_CB_EA_kernel<<< dimGridImg, dimBlockImg>>>(ri, pvdViews[n], dev_idImage, dev_imgArray, \
			-sinf(pvdViews[n].fAngle),-cosf(pvdViews[n].fAngle), nImagePitch, fFOVRadiusSquare);
	}

//	checkCudaErrors( sdkStopTimer( timerInter));
//   	gpuTime_InterCalc= sdkGetTimerValue( timerInter );
//    cout << "|=> Total processing time in GPU: " << gpuTime_InterCalc/1000 << "(s)"<< endl;	
	
	checkCudaErrors( cudaMemcpy( pfData, dev_imgArray, nImageSize, cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaFree(dev_imgArray) );
	SAFE_DELETE(pfProjection_);
	//checkCudaErrors(cudaFreeHost(pfProjection_)); 
	//checkCudaErrors(cudaFreeHost(pvdViews)); 
	checkCudaErrors( cudaFreeArray(cu_sinoOneView) );
}

inline void SetupPrjTexture(cudaArray* cu_imgArray)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    // set texture parameters
    prjTexture3D.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
    prjTexture3D.addressMode[1] = cudaAddressModeClamp;
    prjTexture3D.addressMode[2] = cudaAddressModeClamp;
    prjTexture3D.normalized = false;                      // access with un-normalized texture coordinates
    prjTexture3D.filterMode = cudaFilterModeLinear;      // linear interpolation
	
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(prjTexture3D, cu_imgArray, channelDesc));
}


