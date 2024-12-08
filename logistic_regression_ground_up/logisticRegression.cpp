#include <iostream>
#include "logisticRegression.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

int checks(const std::vector<std::vector<double> >& X_Train, const std::vector<double>& Y_Train) {

    std::size_t YSize = Y_Train.size();
    std::size_t XDim = X_Train.size();

    // Initial dimentionality and binomial checks
    for (std::size_t i=0; i < XDim; ++i) {
        if (YSize != X_Train[i].size()){
            throw std::runtime_error("Error : Input and Output Features are of different sizes");
        }
    }

    for (std::size_t target_val : Y_Train) {
        if (target_val != 0 && target_val != 1) {
            throw std::runtime_error("Error : Target variable can only be binary (0 or 1)");
        }
    }
    
    return 0;
}

std::vector<double> calculate_linear_output (const std::vector<std::vector<double> >& X_Train, const std::vector<double>& W) {
    std::size_t RecordSize = X_Train[0].size();
    std::size_t FieldSize = X_Train.size();

    // Declaring the output variable where the output gets saved to
    std::vector<double> Z(RecordSize, 0.0);

    for (std::size_t records = 0; records < RecordSize; ++records) {
        for (std::size_t fields = 0; fields < FieldSize; ++fields) {
            Z[records] += X_Train[fields][records] * W[fields];
        }

    }

    return Z;
}

int sigmoid_function (std::vector<double>& Z) {

    for (std::size_t outs = 0; outs < Z.size(); ++outs) {
        Z[outs] = 1 / (1 + std::exp(-(Z[outs])));
    }

    return 0;
}

double log_loss (const std::vector<double>& Z, const std::vector<double>& Actual) {
    std::size_t num_elements = Z.size();
    double loss = 0.0;

    for (std::size_t outs = 0; outs < num_elements; ++outs) {

        double actual_outcome = Actual[outs];
        double predicted_prob = Z[outs];
        loss += - ((actual_outcome * std::log(predicted_prob)) + ((1.0 - actual_outcome) * std::log(1.0 - predicted_prob)));
    }

    return loss / num_elements;
}

std::vector<double> calculate_gradient (const std::size_t& Xdim, const std::size_t& YSize, const std::vector<double> Z, const std::vector<std::vector<double> >& X, const std::vector<double>& Y) {

    std::vector<double> gradients(Xdim, 0.0);

    for (std::size_t records = 0; records < YSize; ++records) {
        for (std::size_t fields = 0; fields < Xdim; ++fields) {
            gradients[fields] += ((Z[records] - Y[records]) * X[fields][records]) / YSize;
        }
    }

    return gradients;
}

int update_weights (double& alpha,std::vector<double>& W, const std::vector<double>& gradient) {

    for (std::size_t weight = 0; weight < W.size(); ++weight) {
        W[weight] -= alpha * gradient[weight];
    }

    return 0;
}

int LogisticRegression::fit(const std::vector<std::vector<double> >& X_Train, const std::vector<double>& Y_Train, double learning_rate, std::size_t epochs) {

    std::size_t YSize = Y_Train.size();
    std::size_t XDim = X_Train.size();

    // Initial dimentionality and binomial checks
    checks(X_Train,Y_Train);
    // Print dimensions of X_Train
    std::cout << "X_Train dimensions: " << XDim << "x" << (X_Train.empty() ? 0 : X_Train[0].size()) << std::endl;
    // Print number of elements in Y_Train
    std::cout << "Y_Train size: " << YSize << std::endl;


    // Initializing the weights for linear function
    std::vector<double> w(XDim, 0.5);


    for (std::size_t iter = 0; iter < epochs; ++iter) {
        // Calculate the linear estimator value (z = summation(x*w))
        std::vector<double> Z(YSize, 0.0);
        Z = calculate_linear_output(X_Train,w);
        sigmoid_function(Z);

        // Calculate the gradient of loss function
        std::vector<double> gradient(XDim, 0.0);
        gradient = calculate_gradient(XDim,YSize,Z,X_Train,Y_Train);

        // Update weights
        update_weights(learning_rate, w, gradient);

        // Calculate the log loss (Binary cross entropy)
        double error = 0.0;
        error = log_loss(Z,Y_Train);

        std::cout << "Epoch: " << iter + 1 << ", Error: " << error << ", Weights: ";
        for (const auto& weight : w) {
            std::cout << weight << " ";
        }
        std::cout << ", Gradient: ";
        for (const auto& grad : gradient) {
            std::cout << grad << " ";
        }
        std::cout << std::endl;
    }
    

    return 0;
}
