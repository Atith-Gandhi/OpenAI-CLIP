import numpy as np
import matplotlib.pyplot as plt

sampling_functions = [ 'big_small_sampling', 
                       'supernet_subnet_sampling', 
                       'randomized_sampling', 
                       'no_sampling',
                       'depth_balanced_sampling',
                       'width_balanced_sampling',
                       'ss_bb_sampling'
                    ]

for sf in sampling_functions:
    merged_matrix = np.array([])
    for i in range(9):
        matrix = np.load(f'{sf}_{12*i}.npy') 
        print(matrix.shape)
        merged_matrix = np.concatenate((merged_matrix, matrix))
        print(merged_matrix.shape)
    np.save(f'{sf}_merged_matrix.npy', merged_matrix)

no_sampling_cosine_similarities = np.load('no_sampling_merged_matrix.npy')
randomized_sampling_cosine_similarities = np.load('randomized_sampling_merged_matrix.npy')
big_small_sampling_cosine_similarities = np.load('big_small_sampling_merged_matrix.npy')
supernet_subnet_sampling_cosine_similarities = np.load('supernet_subnet_sampling_merged_matrix.npy')
depth_balanced_sampling_cosine_similarities = np.load('depth_balanced_sampling_merged_matrix.npy')
width_balanced_sampling_cosine_similarities = np.load('width_balanced_sampling_merged_matrix.npy')
ss_bb_sampling_cosine_similarities = np.load('ss_bb_sampling_merged_matrix.npy')

print(np.max(big_small_sampling_cosine_similarities))

# Create the plot with 5 lines
x = range(108)
plt.figure(figsize=(8, 6))  # Set the figure size

# Plot the lines
plt.plot(x, no_sampling_cosine_similarities, linestyle='-', label='No Sampling')
plt.plot(x, randomized_sampling_cosine_similarities, linestyle='-', label='Big Small Sampling')
plt.plot(x, big_small_sampling_cosine_similarities, linestyle='-', label='Randomized Sampling')
plt.plot(x, supernet_subnet_sampling_cosine_similarities, linestyle='-', label='Supernet Subnet Sampling')
plt.plot(x, depth_balanced_sampling_cosine_similarities, linestyle='-', label='Depth Balanced Sampling')
plt.plot(x, width_balanced_sampling_cosine_similarities, linestyle='-', label='Width Balanced Sampling')
plt.plot(x, ss_bb_sampling_cosine_similarities, linestyle='-', label='Small Small, Big Big Sampling')

# Add labels, title, and legend
plt.xlabel('Subnet')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity values for different Subnet')
plt.legend()

# Show the plot
plt.grid(True)  # Add grid lines
plt.savefig("Cosine_Similarity.png")