import numpy as np
import tensorflow as tf
import optuna
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from Phase_retrival import reconTimeSeries
from sdof import integrate, peaks, spectrum
from pandas import DataFrame
from scipy.signal import istft, stft
from tqdm import tqdm
import response_spectrum as sp
import datetime
import os
import sys
from tee import StdoutTee


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def visualize_seismic(data, title="Seismic Data", ylabel="Acceleration (g)", file_name="seismic_data.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Seismic Data')

    # Find the maximum absolute value and its index
    max_value = np.max(np.abs(data))
    max_index = np.argmax(np.abs(data))

    # Add a dotted line at the maximum absolute value
    plt.axvline(x=max_index, color='gray', linestyle='--', label=f'Max Value: {max_value:.2f}')

    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def calculate_ssd(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays must be of the same length.")

    diff = array1 - array2
    ssd = np.sum(diff ** 2)
    return ssd


def record_data_to_file(ground_motion_acceleration, u, ground_motion_force, file_path):
    response_spectrum = sp.get_ReSpe(gm=ground_motion_force)
    np.savez(file_path, ground_motion_acceleration=ground_motion_acceleration, displacement=u,
             response_spectrum=response_spectrum)


def ensure_directory_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def generate_filename(folder, label, extension="png"):
    return os.path.join(folder, f"{label}.{extension}")

def generate_filename_wo_ext(folder, label):
    return os.path.join(folder, f"{label}")


def analyze_sdof_response(acceleration, dt, k, c, m, g, j, folder_path, file_name):
    # Calculate the optimized force using the mass
    optimized_force = acceleration * m
    # Integrate SDOF system response to get displacement, velocity, and acceleration
    u, v, a = integrate(optimized_force, dt, k, c, m)
    # Create a DataFrame for displacement
    u_df = DataFrame(u, columns=['Displacement'])

    # Plot the displacement results
    plt.figure(figsize=(6, 6))
    plt.plot(u_df, color='blue')
    max_value = np.max(np.abs(u_df['Displacement']))
    max_index = np.argmax(np.abs(u_df['Displacement']))
    plt.axvline(x=max_index, color='gray', linestyle='--', label=f'Max Value: {max_value:.5f}')
    plt.title(f"Displacement_{g}_{j}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Displacement (m)')
    plt.legend()
    plt.tight_layout()

    # Save the plot to the specified file
    plt.savefig(file_name)
    plt.close()


def plot_response_spectrum(optimized_force, folder_path, g, j, file_name):
    #
    response_spectrum_diagram = generate_filename(folder_path, f"response_spectrum_{g}_{j}")
    sta = sp.DRS()
    rs1 = sp.get_ReSpe(gm=optimized_force)  # Assuming optimized_force should be used here
    ssd_value = calculate_ssd(sta, rs1)

    f = np.arange(0.1, 10, 1 / 60)
    t = 1 / f

    plt.figure(figsize=(10, 6))
    plt.plot(t, sta, label='Design spectrum', color='dimgray')
    plt.plot(t, rs1, label='Response Spectrum', color='yellow')
    plt.xlabel('Period (s)')
    plt.ylabel('Spectral Acceleration (m/s^2)')
    plt.title(f"Compare to Design Spectrum_{g}_{j} (SSD: {ssd_value:.6f})")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def objective(trial, folder_path, low_bound_data_new, up_bound_data_new, decoder, vmax, vmin, reconTimeSeries):
    global j, best_sdof_for_global
    z_sample = np.zeros(26)
    for i in range(26):
        z_sample[i] = trial.suggest_float(f'z_{i}', low_bound_data_new[i], up_bound_data_new[i])
    z_sample = np.expand_dims(z_sample, axis=0)
    z_sample_float = np.float64(z_sample)
    # Decode to generate ground motion STFT
    ground_motion_stft = decoder(z_sample_float, training=False)[0, :, :, 0] * (vmax - vmin) + vmin
    # Convert STFT to time series acceleration signal
    ground_motion_acceleration = reconTimeSeries(ground_motion_stft)
    ground_motion_force = ground_motion_acceleration * m
    # Integrate SDOF system response
    u, v, a = integrate(ground_motion_force, dt, k, c, m)
    max_displacement = np.max(np.abs(u))
    if max_displacement > best_sdof_for_global:
        file_path = generate_filename_wo_ext(folder_path, f"numerical_data_{g}_{j}")
        record_data_to_file(ground_motion_acceleration, u, ground_motion_force, file_path)
        best_sdof_for_global = max_displacement
        ground_motion_file_name = generate_filename(folder_path, f"ground_acceleration_{g}_{j}")
        sdof_file_name = generate_filename(folder_path, f"sdof_{g}_{j}")
        spectrum_file_name = generate_filename(folder_path, f"spectrum_{g}_{j}")
        visualize_seismic(ground_motion_acceleration, title=f"Optimized Seismic Acceleration_{g}_{j}", ylabel="Acceleration (g)",
                          file_name=ground_motion_file_name)
        # Assuming analyze_sdof_response parameters are ready
        analyze_sdof_response(ground_motion_acceleration, dt, k, c, m, g, j, folder_path, sdof_file_name)
        # plot_response_spectrum(ground_motion_acceleration, folder_path, g, j, spectrum_file_name)
    j = j + 1

    return max_displacement


encoder = load_model('models/VAE/stft_test36_encoder.h5', custom_objects={'Sampling': Sampling})
decoder = load_model('models/VAE/stft_test36_decoder.h5', custom_objects={'Sampling': Sampling})
train_data = np.load('inputs/X_stft_spec_scaled_full_train.npy', allow_pickle=True)
lower_bounds = np.load('outputs/ls_lower_bounds_all.npy', allow_pickle=True)[0]
lower_bounds = np.array(lower_bounds, dtype=np.float64)
end_bounds = np.load('outputs/ls_upper_bounds_all.npy', allow_pickle=True)[0]
end_bounds = np.array(end_bounds, dtype=np.float64)
delta = (end_bounds - lower_bounds) / 15
upper_bounds = lower_bounds + delta
folder_path = 'outputs/result_sdof_26D_confi95_0823'
ensure_directory_exists(folder_path)

# input data are processed with Min-Max method, record vmax and vmin to recover them to real values
vmax = 2.451472710574681
vmin = 5.10990545904576e-10

# System properties (0.5secs)
k = 2.5  # stiffness in N/m
c = 0.1592  # damping coefficient in kg/s
m = 0.015831  # mass in kg
dt = 0.01  # time step

with StdoutTee('outputs/result_sdof_26D_confi95_0823/output.log'):
    j = 0
    g = 1
    counter = 1
    best_sdof_for_global = 0
    not_optimized_continue = 0
    while np.any(upper_bounds < end_bounds):
        print(f'This is the {counter} iteration.')
        print(f'Upper bounds: {upper_bounds}')
        print(f'Lower bounds: {lower_bounds}')
        j = 0
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(lambda trial: objective(trial, folder_path, lower_bounds, upper_bounds, decoder, vmax, vmin, reconTimeSeries), 50)
        best_sdof_this_iteration = study.best_value
        print(f'Best sdof: {best_sdof_this_iteration}')
        print(f'Best parameters: {study.best_params}')
        lower_bounds = np.array([study.best_trial.params[f'z_{i}'] for i in range(26)]) - delta
        upper_bounds = np.array([study.best_trial.params[f'z_{i}'] for i in range(26)]) + delta
        g += 1
        counter += 1
        if not_optimized_continue == 10:
            break
        elif best_sdof_for_global != best_sdof_this_iteration:
            not_optimized_continue += 1
        else:
            not_optimized_continue = 0
