import os
import numpy as np
import matplotlib.pyplot as plt

def load_numpy_arrays_from_folder(folder_path):
    files = os.listdir(folder_path)
    npy_files = [f for f in files if f.endswith('.npy')]
    numpy_arrays = {}
    for npy_file in npy_files:
        file_path = os.path.join(folder_path, npy_file)
        array_name = os.path.splitext(npy_file)[0]
        numpy_arrays[array_name] = np.load(file_path)
    return numpy_arrays

def filter_and_arrays(numpy_dict):
    total_timesteps = '10000000'
    mean_arrays = {name: array for name, array in numpy_dict.items() if name.startswith('mean_rewards_'+total_timesteps+'_')}
    std_arrays = {name: array for name, array in numpy_dict.items() if name.startswith('std_rewards_'+total_timesteps+'_')}
    times_arrays = {name: array for name, array in numpy_dict.items() if name.startswith('training_times_'+total_timesteps+'_')}
    
    return mean_arrays, std_arrays, times_arrays

x_values = np.linspace(100_000, 10_000_000, 100)

name = "breakout"
base_array_path = name + "/A2C_10M/base/arrays"
q3_array_path = name + "/A2C_10M/3/arrays"
numpy_dict_base = load_numpy_arrays_from_folder(base_array_path)
numpy_dict_q3 = load_numpy_arrays_from_folder(q3_array_path)
mean_arrays_base, std_arrays_base, times_arrays_base = filter_and_arrays(numpy_dict_base)
mean_arrays_q3, std_arrays_q3, times_arrays_q3 = filter_and_arrays(numpy_dict_q3)

mean_arrays_base_stack = np.stack(list(mean_arrays_base.values()))
std_arrays_base_stack = np.stack(list(std_arrays_base.values()))
times_arrays_base_stack = np.stack(list(times_arrays_base.values()))

mean_arrays_q3_stack = np.stack(list(mean_arrays_q3.values()))
std_arrays_q3_stack = np.stack(list(std_arrays_q3.values()))
times_arrays_q3_stack = np.stack(list(times_arrays_q3.values()))


plt.figure(1)
plt.plot(x_values, np.mean(mean_arrays_base_stack, axis=0), label='base', linewidth=3, linestyle='--', color='red')
plt.plot(x_values, np.mean(mean_arrays_q3_stack, axis=0), label='q3', linewidth=3, linestyle='--',color = 'green')
plt.title(name +" average performance")
plt.xlabel('Total timesteps')
plt.ylabel('Performance')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x_values, mean_arrays_base_stack[0], label='base', linewidth=2, linestyle='-', color='red')
plt.plot(x_values, mean_arrays_q3_stack[0], label='q3', linewidth=2, linestyle='-',color = 'green')
for i in range(3):
    plt.plot(x_values, mean_arrays_base_stack[i], linewidth=2, linestyle='-', color='red')
    plt.plot(x_values, mean_arrays_q3_stack[i], linewidth=2, linestyle='-',color = 'green')
plt.title(name +" all mean rewards")
plt.xlabel('Total timesteps')
plt.ylabel('Performance')
plt.legend()
plt.show()

print(np.mean(mean_arrays_base_stack, axis=0)[9])
print(np.mean(mean_arrays_q3_stack, axis=0)[9])

print(np.mean(mean_arrays_base_stack, axis=0)[99])
print(np.mean(mean_arrays_q3_stack, axis=0)[99])