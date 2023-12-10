import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

# def identify_pareto(scores):
#     """
#     Identify the Pareto frontier from a set of scores.
#     """
#     pareto_front = np.ones(scores.shape[0], dtype=bool)
#     for i, score in enumerate(scores):
#         if pareto_front[i]:
#             pareto_front[pareto_front] = np.any(scores[pareto_front] >= score, axis=1)
#             pareto_front[i] = True
#     return pareto_front

# # Example scattered points (randomly generated)
# np.random.seed(42)
# points = np.random.rand(20, 2)

# Identify the Pareto frontier


flops = np.load('flops.npy')
randomized_sampling_cosine_similarities = np.load('no_sampling_merged_matrix.npy')
points = np.column_stack((flops, randomized_sampling_cosine_similarities))


# Identify the Pareto front using Convex Hull
hull = ConvexHull(points)
pareto_front_indices = hull.vertices
pareto_front = points[pareto_front_indices]
pareto_front = pareto_front[
    (pareto_front[:, 1] - 0.0001 - np.min(randomized_sampling_cosine_similarities))/(pareto_front[:, 0] + 0.0001 - np.min(flops))
      >= (np.max(randomized_sampling_cosine_similarities) - np.min(randomized_sampling_cosine_similarities))/(np.max(flops) - np.min(flops)) ]

pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
# Plot the scattered data and the Pareto front
plt.scatter(points[:, 0], points[:, 1], label='Points')
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='')
plt.plot(pareto_front[:, 0], pareto_front[:, 1], color='red', linestyle='-', linewidth=2, label='Pareto Front')
plt.xlabel('Latency (ms)')
plt.ylabel('Cosine Similarity')
# plt.ylim(0.2, 0.5)
plt.title('Cosine Similarity vs Latency in No Sampling')
plt.legend()
plt.show()
plt.savefig('No_Balanced_Sampling_Pareto_Frontier_Curve.png')
