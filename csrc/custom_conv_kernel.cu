#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace {

__global__ void conv_forward_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int batch,
                                    int in_channels,
                                    int out_channels,
                                    int height,
                                    int width,
                                    int kernel_h,
                                    int kernel_w,
                                    int stride,
                                    int padding) {
    int n = blockIdx.x;
    int oc = blockIdx.y;
    int y = threadIdx.y;
    int x = threadIdx.x;

    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;

    if (y >= out_h || x >= out_w) return;

    float sum = bias ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int in_y = y * stride + ky - padding;
                int in_x = x * stride + kx - padding;
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int i_idx = n * in_channels * height * width + ic * height * width + in_y * width + in_x;
                    int w_idx = oc * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + ky * kernel_w + kx;
                    sum += input[i_idx] * weight[w_idx];
                }
            }
        }
    }
    int o_idx = n * out_channels * out_h * out_w + oc * out_h * out_w + y * out_w + x;
    output[o_idx] = sum;
}

} // anonymous namespace

std::vector<torch::Tensor> custom_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {

    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);

    auto batch = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);

    auto out_h = (height + 2 * padding - kernel_h) / stride + 1;
    auto out_w = (width + 2 * padding - kernel_w) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_h, out_w}, input.options());

    const int threads = 16;
    dim3 threads_per_block(threads, threads);
    dim3 num_blocks(batch, out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv_forward_cuda", ([&] {
        conv_forward_kernel<<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch,
            in_channels,
            out_channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride,
            padding);
    }));

    return {output};
}
