#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>

using namespace std;

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("AddOne")
    .Attr("preserve_index: int = 0")
    .Input("to_add: int32")
    .Output("added: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    
    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Add one
    cout << "Adding 1" << endl;
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output(i) = input(i) + 1;
    }

  }

};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU), AddOneOp);