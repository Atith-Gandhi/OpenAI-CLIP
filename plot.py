import matplotlib.pyplot as plt
import numpy as np

# def get_seed_average_accuracy(model_name, no_seeds = 5, context = 10):

#     seed_accuracy_array = np.zeros((no_seeds, context, context))

#     # for i in range()
#     # array1 = 
#     for seed in range(1, no_seeds+1):
#         seed_accuracy_array[seed-1] =np.load(f'{model_name}_Context={context}_seed={seed}.npy')

#     return np.mean(seed_accuracy_array, axis=0)


no_sampling_cosine_similarities = np.load('no_sampling_merged_matrix.npy')
randomized_sampling_cosine_similarities = np.load('randomized_sampling_merged_matrix.npy')
big_small_sampling_cosine_similarities = np.load('big_small_sampling_merged_matrix.npy')

# print(saved_accuracy_BIR)
# print(saved_accuracy_powerlaw.shape)
# print(saved_accuracy_ER.shape)  




# Generate some random data for the lines
x = range(108)  # X-axis values
# print(x[1])

# y_powerlaw = []
# y_ER = []
# y_BIR = []
# y_EWC = []
# y_LWF = []
# y_AGEM = []
# for i in range(10):
#     y_powerlaw.append(saved_accuracy_powerlaw[i][i])
#     y_ER.append(saved_accuracy_ER[i][i])
#     y_BIR.append(saved_accuracy_BIR[i][i])
#     y_EWC.append(saved_accuracy_EWC[i][i])
#     y_LWF.append(saved_accuracy_LWF[i][i])
#     y_AGEM.append(saved_accuracy_AGEM[i][i])


# Create the plot with 5 lines
plt.figure(figsize=(8, 6))  # Set the figure size

# Plot the lines
plt.plot(x, no_sampling_cosine_similarities, linestyle='-', label='No Sampling')
plt.plot(x, randomized_sampling_cosine_similarities, linestyle='-', label='Randomized Sampling')
plt.plot(x, big_small_sampling_cosine_similarities, linestyle='-', label='Big Small Sampling')

# Add labels, title, and legend
plt.xlabel('Subnet')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity values for different Subnet')
plt.legend()

# Show the plot
plt.grid(True)  # Add grid lines
plt.savefig("Cosine_Similarity.png")

# saved_accuracy_BIR = np.load("../continual-learning-baseline/save_accuracy_BIR.npy")
# saved_accuracy_EWC = np.load("../continual-learning-baseline/save_accuracy_EWC.npy")
# saved_accuracy_LWF = np.load("../continual-learning-baseline/save_accuracy_LWF.npy")
# saved_accuracy_AGEM = saved_accuracy_LWF = np.load("../continual-learning-baseline/save_accuracy_AGEM_Context=10.npy")