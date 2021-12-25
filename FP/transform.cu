#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "transform.h"
#include<helper_cuda.h>
#include "math.h"
#include <cuda.h>

using namespace std;

#define stepSize 0.125f
#define threadsPerBlockRayTrace 256

texture<float, 2, cudaReadModeElementType> imgTexture2D;

static int _nViews, _nCols, _nX, _nY, _nImageLength, _nProjectionLength, _nSample;
static float _D, _R, _dRange, _dAngle, _rFOV, _dx, _dy, _xmin, _ymin, _dCol, _dLeft, _nViewSize, _nImageSize, _nProjectionSize;
__constant__ int _nViews_, _nCols_, _nX_, _nY_, _nSample_;
__constant__ float _D_, _R_, _dRange_, _dAngle_, _rFOV_, _dx_, _dy_, _xmin_, _ymin_, _dCol_, _dLeft_;

__host__ void
para_setup_fan_curve(float D, float R, int nViews, float dRange, float dAngle, int nCols, int nX, int nY, float rFOV, float dx, float dy, float xmin, float ymin, float dCol, float dLeft, int nSamp)
{

   _D = D;
   _R = R;
   _nViews = nViews;
   _dRange = dRange;
   _dAngle = dAngle;
   _nCols = nCols;
   _nX = nX;
   _nY = nY;
   _rFOV = rFOV;
   _dx = dx;
   _dy = dy;
   _xmin = xmin;
   _ymin = ymin;
   _dCol = dCol;
   _dLeft = dLeft;
   _nViewSize = nCols*sizeof(float);
   _nImageLength = nX*nY;
   _nImageSize = nX*nY*sizeof(float);
   _nProjectionLength = nCols*nViews;
   _nProjectionSize = nCols*nViews*sizeof(float);
   _nSample = nSamp;

   checkCudaErrors( cudaMemcpyToSymbol(_D_, &D, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_R_, &R, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_nViews_, &nViews, sizeof(int) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_dRange_, &dRange, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_dAngle_, &dAngle, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_nCols_, &nCols, sizeof(int) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_nX_, &nX, sizeof(int) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_nY_, &nY, sizeof(int) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_rFOV_, &rFOV, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_dx_, &dx, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_dy_, &dy, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_xmin_, &xmin, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_ymin_, &ymin, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_dCol_, &dCol, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_dLeft_, &dLeft, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_nSample_, &nSamp, sizeof(int) ) );
}

__global__ void
GenerateRayTraceMap_FB_ReconImage_OneView_kernel(float fSourceToCenterDistance, float fAngle, float fColMin, float fDeltaCol, float fFOVRadius, float * pfStart1, float * pfStart2, float * pfStep1, float * pfStep2, int * pnSteps, float * pfDistIntersection)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float fRayAngle = fAngle+fColMin+(i+0.5f)*fDeltaCol;
    float fSinAngle = sinf(fRayAngle);
    float fCosAngle = cosf(fRayAngle);

    float fDistanceToCenterRay = fSourceToCenterDistance*sinf(fColMin+(i+0.5f)*fDeltaCol);

    float intersection1[2],intersection2[2];

    float fDistIntersection = 0;
    int numSamples;


    float x1,x2,y1,y2;
    x1 = -sqrtf(fFOVRadius*fFOVRadius-fDistanceToCenterRay*fDistanceToCenterRay);
    x2 = -x1;
    y1 = -fDistanceToCenterRay;
    y2 = -fDistanceToCenterRay;

    intersection1[0] = x1*fCosAngle-y1*fSinAngle;
    intersection1[1] = x1*fSinAngle+y1*fCosAngle;
    intersection2[0] = x2*fCosAngle-y2*fSinAngle;
    intersection2[1] = x2*fSinAngle+y2*fCosAngle;

    fDistIntersection = (float)sqrt((intersection1[0]-intersection2[0])*(intersection1[0]-intersection2[0])+(intersection1[1]-intersection2[1])*(intersection1[1]-intersection2[1]));
    numSamples	= int(floor(fDistIntersection/stepSize));

    pnSteps[i] = numSamples;
    pfStep1[i] = (intersection2[0]-intersection1[0])/numSamples;
    pfStep2[i] = (intersection2[1]-intersection1[1])/numSamples;
    pfStart1[i] = intersection1[0]+fFOVRadius;
    pfStart2[i] = intersection1[1]+fFOVRadius;
//	pfDistIntersection[i] = (fDistIntersection<1 ? 1: fDistIntersection);

}

__global__ void
RayTrace_ReconImage_OneView_kernel(float * pfPrj, const float * pfStart1, const float * pfStart2, const float * pfStep1, const float * pfStep2, const int * pnSteps)
{
    unsigned int ni=threadIdx.x;
    unsigned int nCacheIndex=ni;
    unsigned int nCol=blockIdx.x;

    __shared__	float pfPrjCache[threadsPerBlockRayTrace];
    pfPrjCache[nCacheIndex] = 0;

    __shared__ int nLength;
    __shared__ float fStart1;
    __shared__ float fStart2;
    __shared__ float fStep1;
    __shared__ float fStep2;

    nLength=pnSteps[nCol];
    fStart1=pfStart1[nCol];
    fStart2=pfStart2[nCol];
    fStep1=pfStep1[nCol];
    fStep2=pfStep2[nCol];

    while(ni<nLength)
    {
        pfPrjCache[nCacheIndex] += tex2D(imgTexture2D, fStart1+ni*fStep1, fStart2+ni*fStep2);
        ni+=threadsPerBlockRayTrace;
    }
    __syncthreads();
    int i = threadsPerBlockRayTrace/2;
    while ( i!= 0)
    {
        if (nCacheIndex < i)
        {
            pfPrjCache[nCacheIndex] += pfPrjCache[nCacheIndex+i];
        }
        __syncthreads();
        i /= 2;
    }

    if(nCacheIndex == 0)
    {
        pfPrj[nCol] = pfPrjCache[0]*stepSize*_dx_;
    }
}

__host__ void
project_fan_curve_gpu(const float * dev_u, float * dev_v, const float * d_pfStart_1, const float * d_pfStart_2, const float * d_pfStep_1, const float * d_pfStep_2, const int * d_pnSteps, cudaArray* cu_ReconImage)
{
    dim3 dimThreadsPerBlockSino(32, 1, 1);
    dim3 dimNumBlocksSino(ceil(_nCols/dimThreadsPerBlockSino.x)+1, 1, 1);
    dim3 dimThreadsPerBlockImg(32, 32, 1);
    dim3 dimNumBlocksImg(_nX / dimThreadsPerBlockImg.x, _nY / dimThreadsPerBlockImg.y, 1);

    float * dev_ybuffer = NULL;
    checkCudaErrors(cudaMalloc((void**) &dev_ybuffer, _nViewSize));
    checkCudaErrors(cudaMemset(dev_ybuffer, 0, _nViewSize));
    checkCudaErrors(cudaMemcpyToArray(cu_ReconImage, 0, 0, dev_u, _nImageSize, cudaMemcpyDeviceToDevice));

    int nV, nOffset;
    for(nV=0; nV<_nViews; nV++)
    {
        nOffset = nV*_nCols;
        RayTrace_ReconImage_OneView_kernel<<< _nCols, threadsPerBlockRayTrace>>>
                (dev_ybuffer, d_pfStart_1+nOffset, d_pfStart_2+nOffset, d_pfStep_1+nOffset, d_pfStep_2+nOffset, d_pnSteps+nOffset);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors( cudaMemcpy( dev_v + nOffset, dev_ybuffer, _nViewSize, cudaMemcpyDeviceToDevice ) );
    }
    checkCudaErrors(cudaFree(dev_ybuffer));
}

__host__ void
project_fan_curve(const float * image, float * projection, const float * anglePos)
{

    ///////////////////////////////////////////////////////////////////////////////////////////////// CUDA Paralleling parameters //////////////////////////////////////////////////////
    dim3 dimThreadsPerBlockSino(32, 1, 1);
    dim3 dimNumBlocksSino(ceil(_nCols/dimThreadsPerBlockSino.x)+1, 1, 1);
    dim3 dimThreadsPerBlockImg(32, 32, 1);
    dim3 dimNumBlocksImg(_nX / dimThreadsPerBlockImg.x, _nY / dimThreadsPerBlockImg.y, 1);
    dim3 dimThreadsPerBlock(32, 32, 1);                                                                                // Define 2D blocks and grids for matrix computation
    dim3 dimBlocksPerGrid(_nX / dimThreadsPerBlock.x, _nY / dimThreadsPerBlock.y, 1);

    ///////////////////////////////////////////////////////////////////////////////////////////// Initialization in Graphic card //////////////////////////////////////////////////////////////////////////
    float * dev_y = NULL;
    float * d_pfDistIntersections = NULL;
    float * d_pfStart_1 = NULL;
    float * d_pfStart_2 = NULL;
    float * d_pfStep_1 = NULL;
    float * d_pfStep_2 = NULL;
    int * d_pnSteps = NULL;
    float * dev_x = NULL;
    float * dev_u = NULL;
    float * dev_v = NULL;

    //////////////////////////////////////////////////////////////////////////////////////////////////////// Allocate Memory in the Graphic card ///////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaMalloc((void**) &dev_y, _nProjectionSize*_nSample));
    checkCudaErrors(cudaMemset(dev_y, 0, _nProjectionSize*_nSample));
    checkCudaErrors(cudaMalloc((void**) &d_pfDistIntersections, _nProjectionSize));
    checkCudaErrors(cudaMemset(d_pfDistIntersections, 0, _nProjectionSize));
    checkCudaErrors(cudaMalloc((void**) &d_pfStart_1, _nProjectionSize));
    checkCudaErrors(cudaMemset(d_pfStart_1, 0, _nProjectionSize));
    checkCudaErrors(cudaMalloc((void**) &d_pfStart_2, _nProjectionSize));
    checkCudaErrors(cudaMemset(d_pfStart_2, 0, _nProjectionSize));
    checkCudaErrors(cudaMalloc((void**) &d_pfStep_1, _nProjectionSize));
    checkCudaErrors(cudaMemset(d_pfStep_1, 0, _nProjectionSize));
    checkCudaErrors(cudaMalloc((void**) &d_pfStep_2, _nProjectionSize));
    checkCudaErrors(cudaMemset(d_pfStep_2, 0, _nProjectionSize));
    checkCudaErrors(cudaMalloc((void**) &d_pnSteps, _nProjectionSize));
    checkCudaErrors(cudaMemset(d_pnSteps, 0, _nProjectionSize));
    checkCudaErrors(cudaMalloc((void**) &dev_x, _nImageSize*_nSample));
    checkCudaErrors(cudaMemset(dev_x, 0, _nImageSize*_nSample));
    checkCudaErrors(cudaMalloc((void**) &dev_u, _nImageSize));
    checkCudaErrors(cudaMemset(dev_u, 0, _nImageSize));
    checkCudaErrors(cudaMalloc((void**) &dev_v, _nProjectionSize));
    checkCudaErrors(cudaMemset(dev_v, 0, _nProjectionSize));

    ///////////////////////////////////////////////////////////////////////////////////////////////// CUDA Paralleling Texture Mapping Setting  //////////////////////////////////////////////////////
    cudaArray * cu_ReconImage = NULL;
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&cu_ReconImage, &channelDescFloat, _nX, _nY));

    imgTexture2D.addressMode[0] = cudaAddressModeClamp;
    imgTexture2D.addressMode[1] = cudaAddressModeClamp;
    imgTexture2D.normalized = false;
    imgTexture2D.filterMode = cudaFilterModeLinear;
    checkCudaErrors(cudaBindTextureToArray(imgTexture2D, cu_ReconImage, channelDescFloat));

    ////////////////////////////////////////////////////////////////////////////////////// Copy data from host to device ////////////////////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaMemcpy(dev_x, image, _nImageSize*_nSample, cudaMemcpyHostToDevice));

    ///////////////////////////////////////////////////////////////////////////// Perform ForwardProjection ////////////////////////////////////////////////////////////////////
    int nV, nOffset, nS;
    for(nV=0; nV<_nViews; nV++)
    {
        nOffset = nV*_nCols;
        GenerateRayTraceMap_FB_ReconImage_OneView_kernel<<<dimNumBlocksSino, dimThreadsPerBlockSino>>>
            (_R/_dx, anglePos[nV], _dLeft, _dCol, _rFOV/_dx,
            d_pfStart_1+nOffset, d_pfStart_2+nOffset, d_pfStep_1+nOffset,
            d_pfStep_2+nOffset, d_pnSteps+nOffset, d_pfDistIntersections+nOffset);
        checkCudaErrors( cudaDeviceSynchronize() );
    }

    for(nS=0; nS<_nSample; nS++)
    {
        checkCudaErrors( cudaMemcpy(dev_u, dev_x+_nImageLength*nS, _nImageSize, cudaMemcpyDeviceToDevice ) );
        checkCudaErrors(cudaMemset(dev_v, 0, _nProjectionSize));
        project_fan_curve_gpu(dev_u, dev_v, d_pfStart_1, d_pfStart_2, d_pfStep_1, d_pfStep_2, d_pnSteps, cu_ReconImage);
        checkCudaErrors( cudaMemcpy(dev_y+_nProjectionLength*nS, dev_v, _nProjectionSize, cudaMemcpyDeviceToDevice ) );
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////// Copy data from device to host ///////////////////////////////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaMemcpy(projection, dev_y, _nProjectionSize*_nSample, cudaMemcpyDeviceToHost));

    //////////////////////////////////////////////////////////////////////////////////////////////////////// Cleanup Graphic card /////////////////////////////////////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaUnbindTexture(imgTexture2D));
    checkCudaErrors(cudaFreeArray(cu_ReconImage));
    checkCudaErrors(cudaFree(dev_y));
    checkCudaErrors(cudaFree(d_pfDistIntersections));
    checkCudaErrors(cudaFree(d_pfStart_1));
    checkCudaErrors(cudaFree(d_pfStart_2));
    checkCudaErrors(cudaFree(d_pfStep_1));
    checkCudaErrors(cudaFree(d_pfStep_2));
    checkCudaErrors(cudaFree(d_pnSteps));
    checkCudaErrors(cudaFree(dev_x));
    checkCudaErrors(cudaFree(dev_u));
    checkCudaErrors(cudaFree(dev_v));
}


