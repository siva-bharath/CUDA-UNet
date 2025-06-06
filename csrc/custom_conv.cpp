#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> custom_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> custom_conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  if (bias.defined()) CHECK_INPUT(bias);
  return custom_conv_forward_cuda(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &custom_conv_forward, "Custom Conv forward (CUDA)");
}
