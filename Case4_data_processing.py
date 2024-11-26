###STEP1###
import re
file_path_1 = 'output.log'


def parse_parameters_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    iterations = re.findall(r'This is the (\d+) iteration\.\n(.*?)\n', data, re.DOTALL)

    params_dict = {}
    for iteration, param_block in iterations:
        param_lines = re.findall(r"'(.*?)': (.*?),", param_block + ',')
        params_dict[f'iteration{iteration}'] = {key: float(value) for key, value in param_lines}

    return params_dict


def calculate_differences(params):
    keys = sorted(params.keys(), key=lambda x: int(x.replace('iteration', '')))
    differences = {}

    for i in range(len(keys) - 1):
        current_params = params[keys[i]]
        next_params = params[keys[i + 1]]
        differences[keys[i] + ' to ' + keys[i + 1]] = {key: next_params[key] - current_params[key] for key in
                                                       current_params}

    return differences


def save_differences_to_file(differences, output_path):
    with open(output_path, 'w') as file:
        for diff in differences:
            file.write(f"Difference {diff}:\n")
            for param, value in differences[diff].items():
                file.write(f"  {param}: {value}\n")
            file.write("\n")

params = parse_parameters_from_file(file_path)
differences = calculate_differences(params)
save_differences_to_file(differences, output_path)

print(f"Differences saved to {output_path}")

###STEP2###
import numpy as np

def read_vectors(file_path):
    vectors = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                dict_str = line.split(': ', 1)[1].strip()
                if dict_str.endswith(','):
                    dict_str = dict_str[:-1]
                vector_dict = eval(dict_str)
                vector = np.array([vector_dict[key] for key in sorted(vector_dict)])
                vectors.append(vector)
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(e)
    return vectors

file_path = 'replace.txt'
vectors = read_vectors(file_path)
print(vectors)
np.save('vectors.npy', vectors)


###STEP3###
import numpy as np

loaded_data = np.load('para_of_iterations.npy')
print(loaded_data)

differences = []

for i in range(len(loaded_data) - 1):
    diff = loaded_data[i + 1] - loaded_data[i]
    differences.append(diff)

differences = np.array(differences)

print(differences)
np.save('vector.npy', differences)


###STEP4###
import numpy as np
import matplotlib.pyplot as plt

loaded_data = np.load('vector.npy')
cosine_values = []
for i in range(len(loaded_data) - 1):
    v1 = loaded_data[i]
    v2 = loaded_data[i + 1]
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        cos_theta = np.nan  # Avoid division by zero
    else:
        cos_theta = dot_product / (norm_v1 * norm_v2)
    cosine_values.append(cos_theta)

plt.figure(figsize=(10, 5))
plt.plot(cosine_values, marker='o', linestyle='-')
plt.title('Cosine of Angles Between Vectors')
plt.xlabel('Index')
plt.ylabel('Cosine Value')
plt.grid(True)
plt.show()