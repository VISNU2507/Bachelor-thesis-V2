# new_rpeak_detector.py

import numpy as np
import scipy.signal as signal

try:
    import neurokit2 as nk
except ImportError:
    raise ImportError("NeuroKit2 is required for this R-peak detection method. "
                      "Please install it with `pip install neurokit2`.")

def detect_r_peaks(ecg_signal, sampling_rate):
    """
    Detect R-peaks in a raw ECG signal using a robust gradient-based
    method from NeuroKit2 (non Pan-Tompkins).

    Parameters:
    - ecg_signal (array-like): 1D array of raw ECG voltage values.
    - sampling_rate (float): Sampling frequency of the ECG signal in Hz.

    Returns:
    - r_peaks (list of int): Indices of detected R-peaks in the ECG signal.
    """
    ecg = np.array(ecg_signal, dtype=float)

    # Step 1: Bandpass filter (5-40 Hz) to clean the ECG signal
    lowcut = 5.0
    highcut = 40.0
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq

    # 4th-order Butterworth bandpass filter
    b, a = signal.butter(N=4, Wn=[low, high], btype='band')
    ecg_filtered = signal.filtfilt(b, a, ecg)

    # Step 2: Use NeuroKit2 to detect R-peaks
    try:
        _, peak_info = nk.ecg_peaks(ecg_filtered, sampling_rate=sampling_rate, method="neurokit")
    except Exception as e:
        raise RuntimeError(f"NeuroKit ECG peak detection failed: {e}")

    r_peaks_indices = peak_info["ECG_R_Peaks"]  # numpy array
    r_peaks = [int(i) for i in r_peaks_indices]
    print(f"Detected {len(r_peaks)} R-peaks with neurokit2.")
    return np.array(r_peaks)



def get_RRI_from_signal(signal, fs, RRI_fs=10):
    # Original:
    # R_peaks = detect_ecg_peaks(signal, fs)

    # New:
    from new_rpeak_detector import detect_r_peaks
    R_peaks_new_2 = detect_r_peaks(signal, fs)

    RRI_new = np.diff(R_peaks_new_2) / fs
    
    return R_peaks_new_2, RRI_new
