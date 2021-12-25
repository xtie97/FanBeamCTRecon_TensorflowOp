#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <math.h>
#include "transform.h"
#include <assert.h>

using namespace tensorflow;
using shape_inference::ShapeHandle;
using std::cerr;
using std::endl;

//////////// TF-OPs for fan-beam curve detector ////////////
REGISTER_OP("BPFanCurve")
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
        auto status = c->GetAttr( "vol_shape", &sp );
        status.Update( c->MakeShapeFromShapeProto( sp, &sh ) );
        c->set_output( 0, sh );
        return Status::OK();
   } )
;

class BPFanCurveOp : public OpKernel
{
protected:
    int nViews, nCols, nX, nY, nZ, nSamp, nv, nc, ns;
    float D, R, dAngle, dRange, rFOV, dx, dy;
    float xmin, ymin, zmin, dCol, dShift, dLeft;
    float fDetBottom, dRow, fstartangle, dz;
    TensorShape projection_shape, volume_shape;
    Eigen::Tensor<float, 1> anglePos;

public:
    explicit BPFanCurveOp(OpKernelConstruction* context) : OpKernel(context) {

        OP_REQUIRES_OK( context, context->GetAttr( "vol_shape", &volume_shape ) );
        OP_REQUIRES_OK( context, context->GetAttr( "proj_shape", &projection_shape ) );

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
        dShift = parameter(9); 
        dLeft = parameter(10)*PI/180.0f;//parameter(10)*PI/180.0f;
        nSamp =  int(parameter(11)); //int(parameter(11));
        fstartangle = parameter(12)*PI/180.0f;//parameter(12)*PI/180.0f; 
        
        //////////////////////////////////////
        para_setup_fan_curve(D, R, nViews, dRange, dAngle, nCols, nX, nY, rFOV, dx, dy, xmin, ymin, dCol, dShift, dLeft, nSamp);
        Tensor angle_tensor;
        OP_REQUIRES_OK( context, context->GetAttr( "angle", &angle_tensor ) );
        auto angle = angle_tensor.tensor<float, 1>();

        anglePos = Eigen::Tensor<float, 1>(nViews);

        for(nv=0; nv<nViews; ++nv)
        {
            anglePos(nv) = angle(nv);
        }

   }

   void Compute(OpKernelContext* context) override
   {
       // check the number of inputs:
       DCHECK_EQ(1, context->num_inputs());

       // import input tensots:
       const Tensor & proj_tensor = context->input(0);

       // check dimensions:
       const TensorShape & proj_shape = proj_tensor.shape();

       DCHECK_EQ(proj_shape.dims(), 3);
       //DCHECK_EQ(vol_shape.dim_size(0), volume_shape(0));

       // get the corresponding Eigen tensors for data access:
       const auto projection = proj_tensor.tensor<float, 3>();

       // create output
       Tensor * vol_tensor = NULL;
       OP_REQUIRES_OK(context, context->allocate_output(0, volume_shape, &vol_tensor));
       auto volume = vol_tensor->tensor<float, 3>();

       backproject_fan_curve(volume.data(), projection.data(), anglePos.data());

   }

};

REGISTER_KERNEL_BUILDER( Name( "BPFanCurve" ).Device( DEVICE_GPU ), BPFanCurveOp );

