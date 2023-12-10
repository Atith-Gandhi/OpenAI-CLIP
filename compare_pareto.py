import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

flops = np.load('flops.npy')
def get_pareto(file_name):
    
    randomized_sampling_cosine_similarities = np.load(file_name)
    points = np.column_stack((flops, randomized_sampling_cosine_similarities))


    # Identify the Pareto front using Convex Hull
    hull = ConvexHull(points)
    pareto_front_indices = hull.vertices
    pareto_front = points[pareto_front_indices]
    pareto_front = pareto_front[
        (pareto_front[:, 1] - 0.0001 - np.min(randomized_sampling_cosine_similarities))/(pareto_front[:, 0] + 0.0001 - np.min(flops))
        >= (np.max(randomized_sampling_cosine_similarities) - np.min(randomized_sampling_cosine_similarities))/(np.max(flops) - np.min(flops)) ]

    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]

    return pareto_front

sampling_functions = ['randomized_sampling_merged_matrix.npy','ss_bb_sampling_merged_matrix.npy', 'depth_balanced_sampling_merged_matrix.npy', 'width_balanced_sampling_merged_matrix.npy', 'supernet_subnet_sampling_merged_matrix.npy', 'no_sampling_merged_matrix.npy' , 'big_small_sampling_merged_matrix.npy']
labels = ['Big Small Sampling', 'Sandwich Sampling', 'Depth Balanced Sampling', 'Width Balanced Sampling', 'Supernet Subnet Sampling', 'No Sampling', 'Randomized Sampling']
point_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan']


for func in sampling_functions:
    pareto_front = get_pareto(func)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], label=labels[sampling_functions.index(func)], color=point_colors[sampling_functions.index(func)])
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], linestyle='-', color=point_colors[sampling_functions.index(func)])

plt.xlabel('Latency (ms)')
plt.ylabel('Cosine Similarity')
# plt.ylim(0.2, 0.5)
plt.title('Paret-Front curve Comparisons')
plt.legend()
plt.show()
plt.savefig('Pareto_Front_Curve_Comparison.png')