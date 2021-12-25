#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "transform.h"
#include<helper_cuda.h>
#include "math.h"
#include <cuda.h>

using namespace std;

#define stepSize 0.125f
#define threadsPerBlockRayTrace 256

texture<float, 1, cudaReadModeElementType> prjTexture1D;

static int _nViews, _nCols, _nX, _nY, _nImageLength, _nProjectionLength, _nSample;
static float _D, _R, _dRange, _dAngle, _rFOV, _dx, _dy, _xmin, _ymin, _dCol, _dShift, _dLeft, _nViewSize, _nImageSize, _nProjectionSize;
__constant__ int _nViews_, _nCols_, _nX_, _nY_, _nSample_;
__constant__ float _D_, _R_, _dRange_, _dAngle_, _rFOV_, _dx_, _dy_, _xmin_, _ymin_, _dCol_, _dShift_, _dLeft_;

__host__ void
para_setup_fan_curve(float D, float R, int nViews, float dRange, float dAngle, int nCols, int nX, int nY, float rFOV, float dx, float dy, float xmin, float ymin, float dCol, float dShift, float dLeft, int nSamp)
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
   _dShift = dShift;
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
   checkCudaErrors( cudaMemcpyToSymbol(_dShift_, &dShift, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_dLeft_, &dLeft, sizeof(float) ) );
   checkCudaErrors( cudaMemcpyToSymbol(_nSample_, &nSamp, sizeof(int) ) );
}


__global__ void
BackProjOneView_fan_curve_kernel(float * pfData, float fSinAngle, float fCosAngle, float fFOVRadiusSquare)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    float fX, fY, fColIndex, fU, fDistanceToCenterRay;
    float * pfVoxel;

    if(i<_nX_ && j<_nY_)
    {
        fX = _xmin_ + _dx_*(i + 0.5f);
        fY = _ymin_ + _dy_*(j + 0.5f);
        if(fX*fX+fY*fY<=fFOVRadiusSquare)
        {
            fU = _R_ + fX*fCosAngle + fY*fSinAngle; //projection of point onto iso-ray
            fDistanceToCenterRay = -fX*fSinAngle + fY*fCosAngle;
            //fOneOverProjDistSquared=1/(fU*fU+fDistanceToCenterRay*fDistanceToCenterRay);
            fColIndex = (atanf(fDistanceToCenterRay/fU) - _dLeft_)/_dCol_ + _dShift_;
            //note the -0.5 is not here as usual because of the texture indexing convention 
            pfVoxel = pfData + j*_nX_ + i;
            if(fColIndex>0 && fColIndex<_nCols_)
            {
                * pfVoxel += tex1D(prjTexture1D, fColIndex);
            }
        }
    }

}

__host__ void
backproject_fan_curve_gpu(float * dev_u, const float * dev_v, const float * angle, cudaArray* cu_sinoOneView)
{
    dim3 dimThreadsPerBlockSino(32, 1, 1);
    dim3 dimNumBlocksSino(ceil(_nCols/dimThreadsPerBlockSino.x)+1, 1, 1);
    dim3 dimThreadsPerBlockImg(32, 32, 1);
    dim3 dimNumBlocksImg(_nX / dimThreadsPerBlockImg.x, _nY / dimThreadsPerBlockImg.y, 1);

    float * dev_ybuffer = NULL;
    checkCudaErrors(cudaMalloc((void**) &dev_ybuffer, _nViewSize));
    checkCudaErrors(cudaMemset(dev_ybuffer, 0, _nViewSize));

    int nV, nOffset;
    for(nV=0; nV<_nViews; nV++)
    {
        nOffset = nV*_nCols;
        checkCudaErrors( cudaMemcpy( dev_ybuffer, dev_v + nOffset, _nViewSize, cudaMemcpyDeviceToDevice ) );
        checkCudaErrors( cudaMemcpyToArray( cu_sinoOneView, 0, 0, dev_ybuffer, _nViewSize, cudaMemcpyDeviceToDevice));
        BackProjOneView_fan_curve_kernel<<< dimNumBlocksImg, dimThreadsPerBlockImg>>>
            (dev_u, -sinf(angle[nV]), -cosf(angle[nV]), _rFOV*_rFOV);
        checkCudaErrors( cudaDeviceSynchronize() );

    }
    checkCudaErrors(cudaFree(dev_ybuffer));
}

__host__ void
backproject_fan_curve(float * image, const float * projection, const float * anglePos)
{

    ///////////////////////////////////////////////////////////////////////////////////////////// Initialization in Graphic card //////////////////////////////////////////////////////////////////////////
    float * dev_y = NULL;
    float * dev_x = NULL;
    float * dev_u = NULL;
    float * dev_v = NULL;

    //////////////////////////////////////////////////////////////////////////////////////////////////////// Allocate Memory in the Graphic card ///////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaMalloc((void**) &dev_y, _nProjectionSize*_nSample));
    checkCudaErrors(cudaMemset(dev_y, 0, _nProjectionSize*_nSample));
    checkCudaErrors(cudaMalloc((void**) &dev_x, _nImageSize*_nSample));
    checkCudaErrors(cudaMemset(dev_x, 0, _nImageSize*_nSample));
    checkCudaErrors(cudaMalloc((void**) &dev_v, _nProjectionSize));
    checkCudaErrors(cudaMemset(dev_v, 0, _nProjectionSize));
    checkCudaErrors(cudaMalloc((void**) &dev_u, _nImageSize));
    checkCudaErrors(cudaMemset(dev_u, 0, _nImageSize));

    ///////////////////////////////////////////////////////////////////////////////////////////////// CUDA Paralleling Texture Mapping Setting  //////////////////////////////////////////////////////
    cudaArray * cu_sinoOneView = NULL;
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&cu_sinoOneView, &channelDescFloat, _nCols, 1));

    prjTexture1D.addressMode[0] = cudaAddressModeClamp;
    prjTexture1D.filterMode = cudaFilterModeLinear;
    prjTexture1D.normalized = false;
    checkCudaErrors(cudaBindTextureToArray(prjTexture1D, cu_sinoOneView, channelDescFloat));

    ////////////////////////////////////////////////////////////////////////////////////// Copy data from host to device ////////////////////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaMemcpy(dev_y, projection, _nProjectionSize*_nSample, cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////// Perform BackProjection ////////////////////////////////////////////////////////////////////////////////////////
    int nS;
    for(nS=0; nS<_nSample; nS++)
    {
        checkCudaErrors( cudaMemcpy(dev_v, dev_y+_nProjectionLength*nS, _nProjectionSize, cudaMemcpyDeviceToDevice ) );
        checkCudaErrors(cudaMemset(dev_u, 0, _nImageSize));
        backproject_fan_curve_gpu(dev_u, dev_v, anglePos, cu_sinoOneView);
        checkCudaErrors( cudaMemcpy(dev_x+_nImageLength*nS, dev_u, _nImageSize, cudaMemcpyDeviceToDevice ) );
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////// Copy data from device to host ///////////////////////////////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaMemcpy(image, dev_x, _nImageSize*_nSample, cudaMemcpyDeviceToHost));

    //////////////////////////////////////////////////////////////////////////////////////////////////////// Cleanup Graphic card /////////////////////////////////////////////////////////////////////////////////////////////////////////
    checkCudaErrors(cudaUnbindTexture(prjTexture1D));
    checkCudaErrors(cudaFreeArray(cu_sinoOneView));
    checkCudaErrors(cudaFree(dev_y));
    checkCudaErrors(cudaFree(dev_x));
    checkCudaErrors(cudaFree(dev_v));
    checkCudaErrors(cudaFree(dev_u));
}


