#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cuda.h>
#include "cuda_util.h"
#include "kernels.h"
#include "llama_kernels.h"
#include "cmath"
#include "memory"
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdexcept>

typedef const torch::Tensor cts;
typedef torch::Tensor ts;

namespace lightseq {
namespace cuda {
template <typename T>
const T *rptr(const torch::Tensor &tensor) {
  return reinterpret_cast<const T *>(tensor.data_ptr());
}

template <typename T>
T *rptr(torch::Tensor &tensor) {
  return reinterpret_cast<T *>(tensor.data_ptr());
}

template <typename T>
void torch_launch_attn_softmax(torch::Tensor &vals,
                               const torch::Tensor &attn_mask, int batch_size,
                               int nhead, int from_len, int to_len,
                               bool is_dec_self_attn, bool mask_future) {
  const T *attn_mask_ptr = rptr<T>(attn_mask);
  if (is_dec_self_attn) {
    attn_mask_ptr = nullptr;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_attn_softmax(rptr<T>(vals), attn_mask_ptr, batch_size, nhead, from_len,
                      to_len, mask_future, stream);
  //     cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}


template <typename T>
void torch_launch_attn_softmax_bw(torch::Tensor &out_grad,
                                  const torch::Tensor &soft_inp, int rows,
                                  int softmax_len) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_attn_softmax_bw(rptr<T>(out_grad), rptr<T>(soft_inp), rows,
                         softmax_len, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_attn_softmax_bw_new(torch::Tensor &inp_grad,
                                      const torch::Tensor &out_grad,
                                      const torch::Tensor &soft_inp, int rows,
                                      int softmax_len) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_attn_softmax_bw_new(rptr<T>(inp_grad), rptr<T>(out_grad),
                             rptr<T>(soft_inp), rows, softmax_len, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

}  // namespace cuda
}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_attn_softmax_fp32",
        &lightseq::cuda::torch_launch_attn_softmax<float>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_fp16",
        &lightseq::cuda::torch_launch_attn_softmax<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_fp32",
        &lightseq::cuda::torch_launch_attn_softmax_bw<float>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_fp16",
        &lightseq::cuda::torch_launch_attn_softmax_bw<__half>,
        "Test kernel wrapper");

  m.def("torch_launch_attn_softmax_bw_new_fp32",
        &lightseq::cuda::torch_launch_attn_softmax_bw_new<float>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_new_fp16",
        &lightseq::cuda::torch_launch_attn_softmax_bw_new<__half>,
        "Test kernel wrapper");
}

