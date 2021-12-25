#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <math.h>
#include <assert.h>
#include "GPURecon_GPU.h"

using namespace tensorflow;
using shape_inference::ShapeHandle;
using std::cerr;
using std::endl;

REGISTER_OP("BackProjection")
    .Input("projection: float")
    .Output("volume: float")
    .Attr("angle: tensor")
    .Attr("parameter: tensor")
    .Attr("vol_shape: shape")
    .Attr("proj_shape: shape")
    .SetShapeFn( []( ::tensorflow::shape_inference::InferenceContext* c )
   {
     shape_inference::ShapeHandle proj_shape;
     TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &proj_shape));

     TensorShapeProto sp;
     ShapeHandle sh;
     auto status = c->GetAttr( "vol_shape", &sp); //volume shape 
     status.Update( c->MakeShapeFromShapeProto(sp, &sh));
     c->set_output( 0, sh );

     return Status::OK();
   });

class BackProjectionOp : public OpKernel
{
protected:
    int nViews, nCols, nX, nY, nZ, nSamp;
    float D, R, dRange, dAngle, rFOV, dx, dy, dz;
    float xmin, ymin, zmin, dCol, dShift, dLeft, fstartangle;
    float fDetBottom, dRow;
    //float *pfProjection_ = NULL;
    //float *pfImage_ = NULL; 
    RotInfo ri;
    TensorShape projection_shape, volume_shape;
    //ViewData * pvdViews;
    ImageData idImage;
    Eigen::Tensor<float, 1> anglePos;

public:
    explicit BackProjectionOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("proj_shape", &projection_shape));
      OP_REQUIRES_OK(context, context->GetAttr("vol_shape", &volume_shape));

      Tensor para_tensor;
      OP_REQUIRES_OK( context, context->GetAttr( "parameter", &para_tensor ) );
      auto parameter = para_tensor.tensor<float, 1>();
      
      D = parameter(0); //parameter(0); //946.73233346;
      R = parameter(1); //parameter(1); //538.52; 
      nViews = int(parameter(2)); //int(parameter(2)); // 1686;
      dRange = parameter(3);//parameter(3); // full scan 360 
      dAngle = dRange/nViews*PI/180.0f;
      nCols = int(parameter(4));//int(parameter(4)); // 888;
      nX = int(parameter(5));//int(parameter(5));
      nY = int(parameter(6));//int(parameter(6));
      rFOV = parameter(7);//parameter(7);
      dx = 2.0f*rFOV/nX;;
      dy = 2.0f*rFOV/nY; 
      xmin = -0.5f*nX*dx;
      ymin = -0.5f*nY*dy;
      dCol = parameter(8)*PI/180.0f; //(parameter(8)/nCols)*PI/180.0f; // 0.061943581081081*PI/180.0f;
      dShift = parameter(9);//parameter(9); //*dCol; //dShift = 1.125; 
      
      //dLeft = -(parameter(8)/2.0f)*PI/180.0f - dShift;
      dLeft = parameter(10)*PI/180.0f;//parameter(10)*PI/180.0f;
      nSamp =  int(parameter(11));
      //////////////////////////////////////
      fstartangle = parameter(12)*PI/180.0f;//parameter(12)*PI/180.0f; 
      fDetBottom = parameter(13);//parameter(13);
      dRow = parameter(14);//parameter(14); // 1.0987828125;
      dz = parameter(15);//parameter(15);  // 2.5;
      nZ = 1; 
      zmin = -0.5f*dz*nZ; //nz=1
 
      //////////////////////////////////////
      ri.fD = D;
      ri.fR = R; 
      ri.nViews = nViews;
      ri.fOffSet = dShift; 
      ri.fStartAngle = fstartangle; 
      ri.fDetLeft = dLeft; 
      ri.fDetBottom = fDetBottom; 
      ri.fDeltaRow = dRow; 
      ri.fDeltaCol = dCol; 
      ri.nRows = 1; 
      ri.nCols = nCols;
      ri.fDeltaAngle = dAngle;
      ri.nSample = nSamp;  
      idImage.nX = nX;
      idImage.nY = nY;
      idImage.nZ = nZ;
      idImage.fXMin = xmin;
      idImage.fYMin = ymin;
      idImage.fZMin = zmin; 
      idImage.fDeltaX = dx;
      idImage.fDeltaY = dy;  
      idImage.fDeltaZ = dz; 
      
      Tensor angle_tensor;
      OP_REQUIRES_OK( context, context->GetAttr( "angle", &angle_tensor ) );
      auto angle = angle_tensor.tensor<float, 1>();

      anglePos = Eigen::Tensor<float, 1>(nViews);

      for(int nv=0; nv<nViews; ++nv)
      {
          anglePos(nv) = angle(nv);
      }

  }
  
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    // check the number of inputs:
    DCHECK_EQ(1, context->num_inputs());
    // import input tensots:
    const Tensor & proj_tensor = context->input(0);
    
    // check dimensions:
    const TensorShape & proj_shape = proj_tensor.shape();
    DCHECK_EQ(proj_shape.dims(), 3);

    // get the corresponding Eigen tensors for data access:
    const auto pfProjection = proj_tensor.tensor<float, 3>(); // pfProjection has a dimension of 4
    
    // create output
    Tensor * vol_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, volume_shape, &vol_tensor));
    auto pfImage = vol_tensor->tensor<float, 3>(); // pfImage has a dimension of 3

	  //////////////////////ApplyCBWeighting(pfProjection_,  pfProjection.data(), ri); 

    //Filtering(pfProjection.data(), pfProjection_, ri);
    //pfImage_  = new float[nX*nY](); 
	  //memcpy(pfImage.data(), pfImage_, nX*nY*sizeof(float));  

    BP_One_Rot_3D_GPU(idImage, pfProjection.data(), pfImage.data(), anglePos.data(), ri, rFOV);

  }
};

REGISTER_KERNEL_BUILDER(Name("BackProjection").Device(DEVICE_GPU), BackProjectionOp);

