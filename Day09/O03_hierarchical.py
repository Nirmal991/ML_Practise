import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

# Generate a synthetic dataset of 20 points
np.random.seed(42)
X = np.random.rand(20, 2)

orignal_dist = pdist(X)

methods = ['ward', 'complete', 'average', 'single']

print("--- Cophenetic Correlation by Linkage Method ---")

for method in methods:
    linked = linkage(X,method=method)
    
    c_coeff, _ = cophenet(linked, orignal_dist)
    
    print(f"Linkage: {method.capitalize():<10} | Cophenetic Score: {c_coeff:.4f}")