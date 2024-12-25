#ifndef ARRAY_UTILITIES_HPP
#define ARRAY_UTILITIES_HPP

#include <iostream>

#include <string>
#include <vector>

class LogisticRegression {
    
    private: 
        std::vector<double> w;
        
    public:
        void set_weights(std::vector<double>& weights) {
            //std::cout << "Instance address: " << this << std::endl;
            w = weights;
        }

        std::vector<double> get_weights() {
            // The below debugging line ensures this function is called using an instance and is not static
            //std::cout << "Instance address: " << this << std::endl;
            return w;
        }

        int fit (const std::vector<std::vector<double> >& X_Train, const std::vector<double>& Y_Train, double learning_rate, std::size_t epochs); 
        std::vector<double> predict (const std::vector< std::vector<double> >& X_Test);//std::vector<double> predict (const std::vector<double>& X_Test);
};

int checks(const std::vector<std::vector<double> >& X_Train, const std::vector<double>& Y_Train);
std::vector<double> calculate_linear_output (const std::vector<std::vector<double> >& X_Train, const std::vector<double>& W);
int sigmoid_function (std::vector<double>& Z);
double log_loss (const std::vector<double>& Z, const std::vector<double>& Actual);
std::vector<double> calculate_gradient (const std::vector<double>& Z, const std::vector<double>& X, const std::vector<double>& Y);
int update_weights (double& alpha,std::vector<double>& W, const std::vector<double>& gradient);

#endif