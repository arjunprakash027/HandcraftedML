from scipy.special import gamma
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# This is how dirichlet is calculated
alpha = [4,1,1]
x = [0.3,0.5,0.2]

alpha_sum = sum(alpha)
numerator = gamma(alpha_sum)
denominator = np.prod([gamma(a) for a in alpha])
normalization = numerator / denominator

print(normalization)
prod_term = 1
for i, (xi,ai) in enumerate(zip(x, alpha)):
  term = xi ** (ai - 1)
  prod_term *= term

pdf = normalization * prod_term
pdf, dirichlet.pdf(x, alpha)

# Visualization example
alpha = [5,1,3]
corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
AREA = 0.5 * 1 * 0.75**0.5
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1e-4):

  coords = np.array([tri_area(xy, pair) for pair in pairs]) / AREA
  bary_coords = np.clip(coords, tol, 1.0 - tol)
  bary_coords_norm = bary_coords / np.sum(bary_coords)
  return bary_coords_norm

refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=2)

# plt.figure(figsize=(8, 4))
# for (i, mesh) in enumerate((triangle, trimesh)):
#     plt.subplot(1, 2, i+ 1)
#     plt.triplot(mesh)
#     plt.axis('off')
#     plt.axis('equal')


pvals = [dirichlet.pdf(xy2bc(xy), alpha) for xy in zip(trimesh.x, trimesh.y)]

print("Sum of pvals", np.sum(pvals))
plt.tricontourf(trimesh, pvals, 200, cmap='jet')
plt.axis('equal')
plt.xlim(0, 1)
plt.ylim(0, 0.75**0.5)
plt.axis('off')
