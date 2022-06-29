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
      .def("get", &FutureReply::get, py::call_guard<py::gil_scoped_release>())
      .def("is_ready", &FutureReply::isReady, py::call_guard<py::gil_scoped_release>())
      .def("is_null", &FutureReply::isNull, py::call_guard<py::gil_scoped_release>());

  py::class_<Batcher>(m, "Batcher")
      .def(py::init<int>())
      .def("exit", &Batcher::exit)
      .def("terminated", &Batcher::terminated)
      .def("send", &Batcher::send, py::call_guard<py::gil_scoped_release>())
      .def("get", &Batcher::get, py::call_guard<py::gil_scoped_release>())
      .def("set", &Batcher::set, py::call_guard<py::gil_scoped_release>());
}
