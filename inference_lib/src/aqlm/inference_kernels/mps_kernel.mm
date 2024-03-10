/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The code that registers a PyTorch custom operation.
*/


#include <torch/extension.h>
#include "CustomSoftshrink.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor& dispatchCode1x16Matvec(
    const torch::Tensor& A,
    const torch::Tensor& B,
          torch::Tensor& C,
    const torch::Tensor& codebook
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of rows.
        int numThreads = C.size(-1);

        // Load the custom soft shrink shader.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string kernel_name = std::string("aqlm_gemv_1x16_kernel_") + "half";
        id<MTLFunction> customSoftShrinkFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(customSoftShrinkFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> softShrinkPSO = [device newComputePipelineStateWithFunction:customSoftShrinkFunction error:&error];
        TORCH_CHECK(softShrinkPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:softShrinkPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(A) offset:A.storage_offset() * A.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(C) offset:C.storage_offset() * C.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(codebook) offset:codebook.storage_offset() * codebook.element_size() atIndex:3];

            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = softShrinkPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numThreads) {
                threadGroupSize = numThreads;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

    return output;
}

// C++ op dispatching the Metal soft shrink shader.
torch::Tensor code1x16_matmat(
    const torch::Tensor& input,
    const torch::Tensor& codes,
    const torch::Tensor& codebooks,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias
) {
    // Check whether the input tensor resides on the MPS device and whether it's contiguous.
    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Check the supported data types for soft shrink.
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Unsupported data type: ", input.scalar_type());

    auto input_sizes = input.sizes();
    auto out_features = codes.size(0) * codebooks.size(2);
    auto flat_input = input.reshape({-1, input.size(-1)});
    auto flat_output = torch::empty({flat_input.size(0), out_features},
        torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device())
    );

    for (int i = 0; i < flat_input.size(0); ++i) {
        auto input_vec = flat_input.index({i});
        auto output_vec = flat_output.index({i});
        dispatchCode1x16Matvec(
            codes.squeeze(2),
            input_vec,
            output_vec,
            codebooks,
            use_bfloat16
        );
    }
    flat_output *= scales.flatten().unsqueeze(0);
    if (bias.has_value()) {
        flat_output += bias->unsqueeze(0);
    }

    auto output_sizes = input_sizes.vec();
    output_sizes.pop_back();
    output_sizes.push_back(-1);
    auto output = flat_output.reshape(output_sizes).clone();
    return output;
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("code1x16_matmat", &code1x16_matmat);
}
