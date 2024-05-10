import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from preprocessing import noiseRemover
import os
from tqdm import tqdm
from ecgdetectors import Detectors
detectors = Detectors(500)

'''
# Detect peaks using a simple thresholding method or find_peaks
def detect_r_peaks(filtered_signal, distance, height=None):
    peaks, _ = signal.find_peaks(abs(filtered_signal), distance=distance, height=height)
    return peaks
'''

# Detect peaks using a simple thresholding method or find_peaks
def detect_r_peaks(unfiltered_signal, distance, height=None):
    r_peaks = detectors.engzee_detector(unfiltered_signal)
    return r_peaks

# Extract QRS complexes from detected peaks
def extract_qrs_complexes(signal, r_peaks, qrs_length=120):
    qrs_complexes = []
    for r_peak in r_peaks:
        start = max(0, r_peak - qrs_length // 2)
        end = min(len(signal), r_peak + qrs_length // 2)
        qrs_complexes.append(signal[start:end])
    return qrs_complexes

# Example function to calculate power of a signal
def calculate_power(signal):
    return np.sqrt(np.mean(signal ** 2))

# Add Gaussian noise to an ECG signal based on QRS energy
def add_gaussian_noise(signal, qrs_power, snr_db=-3):
    # Calculate the noise power needed to reach the desired SNR in dB
    noise_power =  10**(snr_db/20) * qrs_power
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, len(signal))
    noisy_signal = signal + noise
    return noisy_signal

fs = 500
ecg_signal = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold7/normalized-normal-patient1036.npy")[:, 11]
r_peaks = detect_r_peaks(ecg_signal, distance=250)
qrs_complexes = extract_qrs_complexes(ecg_signal, r_peaks, qrs_length=250)
qrs_power = np.mean([calculate_power(qrs) for qrs in qrs_complexes])
noisy_signal = add_gaussian_noise(ecg_signal, qrs_power, snr_db=-3)

plt.figure(figsize=(12, 6))
plt.plot(ecg_signal, linestyle="solid", label='Original ECG Signal')
plt.scatter(r_peaks, ecg_signal[r_peaks], color='red', marker='x', label='Detected R-Peaks')
plt.plot(noisy_signal, linestyle=":", label='Noisy ECG Signal', alpha=0.7)
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('ECG Signal with Detected R-Peaks and Added Gaussian Noise')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
for i, qrs in enumerate(qrs_complexes[:5], 1):
    plt.plot(qrs, label=f'QRS Complex {i}')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Extracted QRS Complexes')
plt.legend()
plt.grid(True)
plt.show()


from PyEMD import EMD, Visualisation

emd = EMD()
emd.emd(ecg_signal)
imfs, res = emd.get_imfs_and_residue()
vis = Visualisation()
t = np.arange(0, 10, 0.002)
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs)
vis.show()

print("testing time")
files = os.listdir("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/")
count = 0
for file in tqdm(files):
    tempArray = np.load(f"/home/tzikos/Desktop/Data/Berts final/tachy/fold1/{file}")
    for i in range(12):
        try:
            # Detect R-peaks
            r_peaks = detect_r_peaks(tempArray[:,i], distance=500)
            # Extract QRS complexes around detected peaks
            qrs_complexes = extract_qrs_complexes(tempArray[:,i], r_peaks, qrs_length=120)
            # Calculate the average power of the QRS complexes
            qrs_power = np.mean([calculate_power(qrs) for qrs in qrs_complexes])
            # Add Gaussian noise with a power 3 dB below the QRS energy
            noisy_signal = add_gaussian_noise(tempArray[:,i], qrs_power, snr_db=-3)
            '''
            plt.figure(figsize=(12, 6))
            plt.plot(tempArray[:,i], linestyle="solid", label='Original ECG Signal')
            plt.scatter(r_peaks, tempArray[:,i][r_peaks], color='red', marker='x', label='Detected R-Peaks')
            plt.plot(noisy_signal, linestyle=":", label='Noisy ECG Signal', alpha=0.7)
            plt.legend()
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title(f'ECG Signal with Detected R-Peaks and Added Gaussian Noiseb Lead {i}')
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(10, 4))
            for i, qrs in enumerate(qrs_complexes, 1):
                plt.plot(qrs, label=f'QRS Complex {i}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Extracted QRS Complexes')
            plt.legend()
            plt.grid(True)
            plt.show()
            '''
            emd = EMD()
            emd.emd(ecg_signal)
            imfs, res = emd.get_imfs_and_residue()
            '''
            vis = Visualisation()
            t = np.arange(0, 10, 0.002)
            vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
            vis.plot_instant_freq(t, imfs=imfs)
            vis.show()
            '''
        except:
            count+=1
            
print(f"{count}/{(len(files)*13)} files failed")
np.save(f"/home/tzikos/Desktop/test.npy", tempArray)
