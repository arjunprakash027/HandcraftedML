#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Required for automatic conversion between Python list and C++ vector
#include "array_utilities.hpp"

PYBIND11_MODULE(example,m) {
    m.doc() = "Vector Addition Module";
    m.def("sumcpp",&array_utilities::sum,"Function to add all values in a array");
}














