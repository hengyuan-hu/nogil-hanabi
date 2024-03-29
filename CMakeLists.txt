CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(pyhanabi)
# to make ABI consistent with pytorch's internal ABI
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi-learning-environment)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11)

# get pybind flags from pytorch so that it is consistent
execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/get_pybind_flags.py
  OUTPUT_VARIABLE PYBIND_FLAGS
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS}")
message(">>>pybind flags<<<")
message(${PYBIND_FLAGS})
message("------------------")

# pybind rla, which depends on pytorch as it uses tensor
# find pytorch
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)
set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")

message(">>>Torch Libs<<<")
message("${TorchPath}")
message("${TORCH_LIBRARIES}")
message("----------------")

pybind11_add_module(rla
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/batcher.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/rla_pybind.cc
)
target_include_directories(rla PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rla PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rla PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rla PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

# pybind hle
pybind11_add_module(hle ${CMAKE_CURRENT_SOURCE_DIR}/cpp/hle_pybind.cc)
target_include_directories(hle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(hle PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(hle PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hle PUBLIC hanabi)
target_link_libraries(hle PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
