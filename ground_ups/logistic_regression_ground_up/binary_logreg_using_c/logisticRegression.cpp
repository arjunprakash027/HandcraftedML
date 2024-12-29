#include <iostream>
#include "logisticRegression.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iomanip>

// There are some initial checks that must be passed before the model is trained such as checking the dimensions of input and output features
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

// Calculating the linear output of the model without applying sigmoid function to it
std::vector<double> calculate_linear_output (const std::vector<std::vector<double> >& X_Train, const std::vector<double>& W) {
    std::size_t RecordSize = X_Train[0].size();
    std::size_t FieldSize = X_Train.size();

    // Declaring the output variable where the output gets saved to
    std::vector<double> Z(RecordSize, 0.0);

    // looping through number of records (data points)
    for (std::size_t records = 0; records < RecordSize; ++records) {
        // Looping through the number of features, since the data is stored in column major format. Lets say we have 10 fields and 1000 records, then the data would look like Data[10 * fields [1000 * records]], shape of 10,1000
        for (std::size_t fields = 0; fields < FieldSize; ++fields) {

            // Every record has an output value, which is sum of all every feature for a particular record multiplied by the weight of that feature
            Z[records] += X_Train[fields][records] * W[fields];
        }

    }

    return Z;
}

// The function that turns linear gibrish output of linear model into probablities 
int sigmoid_function(std::vector<double>& Z) {
    double epsilon = 1e-15;  // Small value for clamping
    for (auto& z : Z) {
        z = 1.0 / (1.0 + std::exp(-z));  // Apply sigmoid
        z = std::max(epsilon, std::min(z, 1.0 - epsilon));  // Clamp output
    }

    return 0;
}

// Logloss is the cost function of logistic regression model that is to be optimized
double log_loss(const std::vector<double>& Z, const std::vector<double>& Actual) {
    std::size_t num_elements = Z.size();
    if (num_elements != Actual.size()) {
        throw std::invalid_argument("Size mismatch between predicted probabilities and actual values.");
    }

    double loss = 0.0;

    for (std::size_t outs = 0; outs < num_elements; ++outs) {
        // Get actual and predicted values
        double actual_outcome = Actual[outs];
        double predicted_prob = Z[outs];

        // Calculate components of the contribution
        double log_predicted = std::log(predicted_prob);
        double log_one_minus_predicted = std::log(1.0 - predicted_prob);
        double actual_part = actual_outcome * log_predicted;
        double negative_part = (1.0 - actual_outcome) * log_one_minus_predicted;

        // Calculate the contribution to the loss
        double contribution = -(actual_part + negative_part);

        // Accumulate the loss
        loss += contribution;

        // Debugging statements for each record
        std::cout << "[DEBUG] Record " << outs + 1 << ":\n";
        std::cout << "        Actual       : " << actual_outcome << "\n";
        std::cout << "        Predicted    : " << predicted_prob << "\n";
        std::cout << "        log(p)       : " << log_predicted << "\n";
        std::cout << "        log(1 - p)   : " << log_one_minus_predicted << "\n";
        std::cout << "        Actual Part (y * log(p)): " << actual_part << "\n";
        std::cout << "        Negative Part ((1 - y) * log(1 - p)): " << negative_part << "\n";
        std::cout << "        Contribution : " << contribution << "\n";
    }

    double average_loss = loss / num_elements;

    // Debugging statement for total loss
    std::cout << "[DEBUG] Total Loss: " << loss << "\n";
    std::cout << "[DEBUG] Average Loss (Log Loss): " << average_loss << "\n";

    return average_loss;
}

// Most important concept in optimization, it calculates the gradient (diffrential of loss function, not calculated here, values are pluged into this diffrentited function to get the gradient) where the loss function moves and tries to minimize the loss function

std::vector<double> calculate_gradient(
    const std::size_t& Xdim,
    const std::size_t& YSize,
    const std::vector<double> Z,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& Y) {

    std::vector<double> gradients(Xdim, 0.0);

    for (std::size_t records = 0; records < YSize; ++records) {
        std::cout << "[DEBUG] Record " << records + 1 << ":\n";
        for (std::size_t fields = 0; fields < Xdim; ++fields) {

            // Gradient computation for each weight
            double gradient_contribution = ((Z[records] - Y[records]) * X[fields][records]) / YSize;
            gradients[fields] += gradient_contribution;

            // Debugging statement for each field
            std::cout << "    Field " << fields + 1 << ":\n";
            std::cout << "        Z[records]: " << Z[records] << "\n";
            std::cout << "        Y[records]: " << Y[records] << "\n";
            std::cout << "        X[fields][records]: " << X[fields][records] << "\n";
            std::cout << "        Contribution to Gradient: " << gradient_contribution << "\n";
            std::cout << "        Updated Gradient[Field]: " << gradients[fields] << "\n";
        }
    }

    // Final gradient debug
    std::cout << "[DEBUG] Final Gradients:\n";
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        std::cout << "    Gradient[" << i + 1 << "]: " << gradients[i] << "\n";
    }

    return gradients;
}


// Updating the weights of model based on gradient and learning rate (alpha)
int update_weights(double& alpha, std::vector<double>& W, const std::vector<double>& gradient) {

    std::cout << "[DEBUG] Updating Weights:\n";
    std::cout << "    Learning Rate (alpha): " << alpha << "\n";

    for (std::size_t weight = 0; weight < W.size(); ++weight) {
        std::cout << "    Weight " << weight + 1 << ":\n";
        std::cout << "        Old Weight: " << W[weight] << "\n";
        std::cout << "        Gradient: " << gradient[weight] << "\n";

        // Update weight
        W[weight] -= alpha * gradient[weight];

        std::cout << "        Updated Weight: " << W[weight] << "\n";
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
    std::vector<double> w(XDim, 0.0);

    for (std::size_t iter = 0; iter < epochs; ++iter) {
        std::cout << "\n=== Debugging Epoch " << iter + 1 << " ===\n";

        // Calculate the linear estimator value (z = summation(x*w))
        std::vector<double> Z(YSize, 0.0);
        std::cout << "[DEBUG] Calculating linear output (Z)...\n";
        Z = calculate_linear_output(X_Train, w);
        std::cout << "[DEBUG] Z (Linear Output): ";
        for (const auto& z : Z) {
            std::cout << std::fixed << std::setprecision(4) << z << " ";
        }
        std::cout << "\n";

        // Apply sigmoid function
        std::cout << "[DEBUG] Applying sigmoid function...\n";
        sigmoid_function(Z);
        std::cout << "[DEBUG] Z after sigmoid: ";
        for (const auto& z : Z) {
            std::cout << std::fixed << std::setprecision(4) << z << " ";
        }
        std::cout << "\n";

        // Calculate the gradient of loss function
        std::vector<double> gradient(XDim, 0.0);
        std::cout << "[DEBUG] Calculating gradient...\n";
        gradient = calculate_gradient(XDim, YSize, Z, X_Train, Y_Train);
        std::cout << "[DEBUG] Gradient: ";
        for (const auto& grad : gradient) {
            std::cout << std::fixed << std::setprecision(4) << grad << " ";
        }
        std::cout << "\n";

        // Update weights
        std::cout << "[DEBUG] Updating weights...\n";
        update_weights(learning_rate, w, gradient);
        std::cout << "[DEBUG] Weights after update: ";
        for (const auto& weight : w) {
            std::cout << std::fixed << std::setprecision(4) << weight << " ";
        }
        std::cout << "\n";

        // Calculate the log loss (Binary cross entropy)
        double error = 0.0;
        std::cout << "[DEBUG] Calculating log loss...\n";
        error = log_loss(Z, Y_Train);
        std::cout << "[DEBUG] Log Loss (Error): " << std::fixed << std::setprecision(6) << error << "\n";

        // Print epoch summary
        std::cout << "\nEpoch Summary:\n";
        std::cout << "Epoch: " << iter + 1 << "\n"
                << "---------------------------\n"
                << "Error      : " << std::fixed << std::setprecision(6) << error << "\n"
                << "Weights    : ";
        for (const auto& weight : w) {
            std::cout << std::fixed << std::setprecision(4) << weight << " ";
        }
        std::cout << "\nGradient   : ";
        for (const auto& grad : gradient) {
            std::cout << std::fixed << std::setprecision(4) << grad << " ";
        }
        std::cout << "\n---------------------------\n";

    }


    // Setting the computed weights so that it could be used for inference
    // The below calling of function is called without creting a instance of the class even tho the declaration is not static
    // this behaviour works in pybind but not in native c++, so please be aware to use static methods or instantiate a object while using normal c++
    LogisticRegression::set_weights(w);

    return 0;
}

std::vector<double> LogisticRegression::predict(const std::vector< std::vector<double> >& X_Test) {
    std::vector<double> w = LogisticRegression::get_weights();
    
    std::size_t XDim = X_Test.size();
    std::size_t WSize = w.size();
    std::size_t RecordSize = X_Test[0].size();

    if (XDim != WSize) {
        throw std::runtime_error("Error : Input and weights are of different sizes, or the weights have not been trained yet! Please check your input or train the model first");
    }

    std::vector<double> Z(RecordSize, 0.0);

    Z = calculate_linear_output(X_Test,w);
    sigmoid_function(Z);

    return Z;   
}