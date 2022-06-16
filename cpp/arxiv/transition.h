// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <torch/extension.h>

#include "tensor_dict.h"

namespace rela {

class RNNTransition {
 public:
  RNNTransition() = default;

  RNNTransition index(int i) const;

  TensorDict toDict();

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;
};

RNNTransition makeBatch(
    const std::vector<RNNTransition>& transitions, const std::string& device);

TensorDict makeBatch(
    const std::vector<TensorDict>& transitions, const std::string& device);

}  // namespace rela
