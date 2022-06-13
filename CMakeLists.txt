CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(pyhanabi)
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) #

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi-learning-environment)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11)

# pybind module
pybind11_add_module(
  hle
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/pybind.cc
)
target_include_directories(hle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hle PRIVATE hanabi)
