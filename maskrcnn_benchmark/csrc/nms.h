// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold) {
  if (dets.numel() == 0) {
//AT_ASSERTM(false, "--------------------------nms_at::empty-------------------------");
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  if (dets.type().is_cuda()) {
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
//AT_ASSERTM(false, "--------------------------nms_cuda-------------------------");
    return nms_cuda(b, threshold);
  }
  else
{
//AT_ASSERTM(false, "--------------------------nms_cpu-------------------------");
  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;

}

  
}
