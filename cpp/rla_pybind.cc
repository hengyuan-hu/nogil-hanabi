// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "cpp/batcher.h"

namespace py = pybind11;
using namespace rla;

PYBIND11_MODULE(rla, m) {
  py::class_<FutureReply>(m, "FutureTensorDict")
      .def(py::init<>())
      .def("get", &FutureReply::get)
      .def("is_ready", &FutureReply::isReady)
      .def("is_null", &FutureReply::isNull);

  py::class_<Batcher>(m, "Batcher")
      .def(py::init<int>())
      .def("exit", &Batcher::exit)
      .def("terminated", &Batcher::terminated)
      .def("send", &Batcher::send)
      .def("get", &Batcher::get)
      .def("set", &Batcher::set);
}
