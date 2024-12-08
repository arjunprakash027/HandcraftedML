#ifndef ARRAY_UTILITIES_HPP
#define ARRAY_UTILITIES_HPP

#include <string>
#include <vector>

class LogisticRegression {
    public:
        static int fit (const std::vector<std::vector<double> >& X_Train, const std::vector<double>& Y_Train, double learning_rate, std::size_t epochs); 
};

int checks(const std::vector<std::vector<double> >& X_Train, const std::vector<double>& Y_Train);
std::vector<double> calculate_linear_output (const std::vector<std::vector<double> >& X_Train, const std::vector<double>& W);
int sigmoid_function (std::vector<double>& Z);
double log_loss (const std::vector<double>& Z, const std::vector<double>& Actual);
std::vector<double> calculate_gradient (const std::vector<double>& Z, const std::vector<double>& X, const std::vector<double>& Y);
int update_weights (double& alpha,std::vector<double>& W, const std::vector<double>& gradient);

#endif