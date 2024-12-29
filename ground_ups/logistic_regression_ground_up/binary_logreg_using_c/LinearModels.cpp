#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Required for automatic conversion between Python list and C++ vector
#include "logisticRegression.hpp"

namespace py = pybind11;

PYBIND11_MODULE(LinearModels,m) {
    m.doc() = "LinearModels Module written in CPP interface in python";

    py::class_<LogisticRegression>(m,"LogisticRegression")
        .def(py::init<>())
        //.def("set_weights",&LogisticRegression::set_weights)
        //.def("get_weights",&LogisticRegression::get_weights)
        .def("fit",&LogisticRegression::fit)
        .def("predict",&LogisticRegression::predict);

    //m.def("logreg_fit",&LogisticRegression::fit,"Function to fit a logistic regression model");
    //m.def("logreg_predict",&LogisticRegression::predict,"Function to predict the output of a logistic regression model");
}














