#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Required for automatic conversion between Python list and C++ vector
#include "logisticRegression.hpp"

PYBIND11_MODULE(LinearModels,m) {
    m.doc() = "LinearModels Module written in CPP interface in python";
    m.def("logreg_fit",&LogisticRegression::fit,"Function to fit a logistic regression model");
}














