#include <iostream>
#include "array_utilities.hpp"
#include <vector>

int array_utilities::sum(const std::vector<int>& vec){
    int total = 0;
    for (size_t i = 0; i < vec.size(); i++){
        total += vec[i];
    }
    return total;
}

