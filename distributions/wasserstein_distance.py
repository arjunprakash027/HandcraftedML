"""
Author : Arjun Prakash
Edit : 14/01/25

A small script to compute 1D wasserstein distance using python
"""

import numpy as np
from scipy.stats import wasserstein_distance

def wasserstein_distance_custom(A, B):
    # Convert inputs to numpy arrays (if not already)
    A = np.array(A)
    B = np.array(B)

    print(f"Input Array A: {A}")
    print(f"Input Array B: {B}")
    print("-" * 50)

    # Step 1: Create a shared sorted axis by merging A and B
    merged = np.sort(np.concatenate([A, B]))
    print(f"Shared Sorted Axis (Merged Array): {merged}")
    print("-" * 50)

    # Step 2: Calculate the CDF for A and B using searchsorted
    # searchsorted counts how many elements in A or B are <= each element in the merged array
    cdf_a = np.searchsorted(A, merged, side='right') / len(A)
    cdf_b = np.searchsorted(B, merged, side='right') / len(B)

    print(f"CDF of A (cumulative probabilities): {cdf_a}")
    print(f"CDF of B (cumulative probabilities): {cdf_b}")
    print("-" * 50)

    # Step 3: Compute the absolute differences between the CDFs
    absolute_difference = np.abs(cdf_a - cdf_b)
    print(f"Absolute Differences between CDFs: {absolute_difference}")
    print("-" * 50)

    # Step 4: Calculate segment lengths between consecutive elements in the merged array
    # Prepend the first element to keep the output length the same as the merged array
    segment_length = np.diff(merged)
    print(f"Segment Lengths (Differences between consecutive merged values): {segment_length}")
    print("-" * 50)

    # Step 5: Compute Wasserstein Distance
    # Sum of (absolute difference * segment length)
    print(absolute_difference[:-1])
    wasserstein_distance = np.sum(absolute_difference[:-1] * segment_length)
    print(f"Wasserstein Distance: {wasserstein_distance}")
    print("=" * 50)

    return wasserstein_distance


if __name__ == '__main__':
    A = [1, 3, 5, 6]
    B = [2, 4, 5, 6]
    wasserstein_distance_custom(A, B)
    print("Scipy WD:",wasserstein_distance(A,B))