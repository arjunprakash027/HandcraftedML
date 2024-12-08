#include <iostream>
#include <vector>
#include "array_utilities.hpp"

int main() {
    std::vector<int> values = {1,4,6,7};
    int result = array_utilities::sum(values);
    std::cout << "Sum value = " << result << std::endl;
    return 0;
}




