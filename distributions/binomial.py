"""
Author : Arjun Prakash
Date : 2024-12-25
Description : To understand how binomial distribution works
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def binomial_distribution(n: int, 
                          p: float, 
                          k:int) -> float:

    nk = math.factorial(n) / math.factorial(k) / math.factorial(n - k)
    pxxk = (p ** k)
    qxxnmk = ((1 - p) ** (n - k))

    return nk * pxxk * qxxnmk

# pass in a binary outcome list to plot the distribution
def plot_binomial_distribution(dist: list) -> None:

    probablity = np.mean(dist)
    n = len(dist)

    binomial_distribution_values = []
    for i in range(len(dist)):
        binomial_distribution_values.append(
                                    binomial_distribution(n = n,
                                    p = probablity,
                                    k = i)
                                    )

    plt.bar(range(len(dist)), binomial_distribution_values)
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distribution for n: {n} and p: {probablity}')
    plt.show()

Y = np.random.randint(0,2,100)

plot_binomial_distribution(Y)