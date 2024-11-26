import numpy as np
import tensorflow as tf
import optuna
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from Phase_retrival import reconTimeSeries
from sdof import integrate, peaks, spectrum
from pandas import DataFrame
from PySeismoSoil.class_ground_motion import Ground_Motion
from scipy.signal import istft, stft
from scipy.signal import istft, stft
from tqdm import tqdm
import response_spectrum as sp
import datetime
import os
from optuna.visualization import plot_optimization_history

class Sampling(Layer):
    """Sampling layer to sample from the normal distribution defined by z_mean and z_log_var."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def visualize_seismic(data, title="Seismic Data", ylabel="Acceleration (g)", file_name="seismic_data.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Seismic Data')
    max_value = np.max(np.abs(data))
    max_index = np.argmax(np.abs(data))
    plt.axvline(x=max_index, color='gray', linestyle='--', label=f'Max Value: {max_value:.2f}')
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(file_name)
    plt.show()


def ensure_directory_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def generate_filename(folder, label, extension="png"):
    return os.path.join(folder, f"{label}.{extension}")


def stop_after_no_improvement(study, trial):
    n_trials = 200  # Set the number of trials after which to stop if no improvement
    if study.best_trial.number + n_trials <= trial.number:
        print(f"No improvement in {n_trials} trials")
        raise optuna.exceptions.OptunaError(f"No improvement in {n_trials} trials")


def analyze_sdof_response(acceleration, dt, k, c, m, j, folder_path):
    # Calculate the optimized force using the mass
    optimized_force = acceleration * m
    # Integrate SDOF system response
    u, v, a = integrate(optimized_force, dt, k, c, m)
    u_df = DataFrame(u, columns=['Displacement'])
    v_df = DataFrame(v, columns=['Velocity'])
    a_df = DataFrame(a, columns=['Acceleration'])
    # SDOF system response visualization
    sdof_file = generate_filename(folder_path, f"sdof_{j}")
    # Plot the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(u_df, color='blue')
    plt.title(f"Displacement_{j}")
    plt.xlabel('Time step')
    plt.ylabel('Displacement (m)')
    plt.subplot(1, 3, 2)
    plt.plot(v_df, color='green')
    plt.title(f"Velocity_{j}")
    plt.xlabel('Time step')
    plt.ylabel('Velocity (m/s)')
    plt.subplot(1, 3, 3)
    plt.plot(a_df, color='red')
    plt.title(f"Acceleration_{j}")
    plt.xlabel('Time step')
    plt.ylabel('Acceleration (m/s^2)')
    plt.tight_layout()
    plt.savefig(sdof_file)

def plot_response_spectrum(optimized_force, folder_path, j):
    #
    response_spectrum_diagram = generate_filename(folder_path, f"response_spectrum_{j}")
    sta = sp.DRS()
    rs1 = sp.get_ReSpe(gm=optimized_force)  # Assuming optimized_force should be used here

    f = np.arange(0.1, 10, 1 / 60)
    t = 1 / f

    plt.figure(figsize=(10, 6))
    plt.plot(t, sta, label='Design spectrum', color='dimgray')
    plt.plot(t, rs1, label='Response Spectrum', color='yellow')
    plt.xlabel('Period (s)')
    plt.ylabel('Spectral Acceleration (m/s^2)')
    plt.title(f"Compare to Design Spectrum_{j}")
    plt.grid(True)
    plt.legend()
    plt.savefig(response_spectrum_diagram)


def objective(trial, folder_path, low_bound_data_new, up_bound_data_new, decoder, vmax, vmin, reconTimeSeries):
    global j, best_pga
    z_sample = np.zeros(26)
    for i in range(26):
        z_sample[i] = trial.suggest_float(f'z_{i}', low_bound_data_new[i], up_bound_data_new[i])
    z_sample = np.expand_dims(z_sample, axis=0)
    z_sample_float = np.float64(z_sample)
    # Decode to generate ground motion STFT
    ground_motion_stft = decoder(z_sample_float, training=False)[0, :, :, 0] * (vmax - vmin) + vmin
    # Convert STFT to time series acceleration signal
    ground_motion_acceleration = reconTimeSeries(ground_motion_stft)
    pga = np.max(np.abs(ground_motion_acceleration))
    # Update and visualize if this is the best PGA found so far for this iteration
    if pga > best_pga:
        best_pga = pga
        ground_motion_file_name = generate_filename(folder_path, f"ground_acceleration_{j}")
        visualize_seismic(ground_motion_acceleration, title=f"Optimized Seismic Acceleration_{j}", ylabel="Acceleration (m/s^2)",
                          file_name=ground_motion_file_name)
        plot_response_spectrum(ground_motion_acceleration, folder_path, j)
        print(f"PGA Optimized!")

    j = j + 1

    return pga


encoder = load_model('models/VAE/stft_test36_encoder.h5', custom_objects={'Sampling': Sampling})
decoder = load_model('models/VAE/stft_test36_decoder.h5', custom_objects={'Sampling': Sampling})
train_data = np.load('inputs/X_stft_spec_scaled_full_train.npy', allow_pickle=True)
up_bound_data = np.load('outputs/(2)Extract_top_bottom_10_stft/ls_upper_bounds_all.npy', allow_pickle=True)[0]
up_bound_data_2 = np.array(up_bound_data, dtype=np.float64)
low_bound_data = np.load('outputs/(2)Extract_top_bottom_10_stft/ls_lower_bounds_all.npy', allow_pickle=True)[0]
low_bound_data_2 = np.array(low_bound_data, dtype=np.float64)
folder_path = 'outputs/(4)Result_26D_confi95'
ensure_directory_exists(folder_path)

# input data are processed with Min-Max method, record vmax and vmin to recover them to real values
vmax = 2.451472710574681
vmin = 5.10990545904576e-10



try:
    j = 0
    best_pga = 0
    decoder = load_model('models/VAE/stft_test36_decoder.h5', custom_objects={'Sampling': Sampling})
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.optimize(lambda trial: objective(trial, folder_path, low_bound_data_2, up_bound_data_2, decoder, vmax, vmin, reconTimeSeries), callbacks=[stop_after_no_improvement])
    fig = plot_optimization_history(study)
    fig.write_html(f"outputs/optimization_history.html")

except:
    pass