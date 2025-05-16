
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class EngzeeTompkins:
    """
    Engzee/Hamilton-based approach for robust R-peak detection in noisy ECG signals.

    1) Bandpass filter the raw ECG to focus on QRS frequency band.
    2) Differentiate to accentuate rapid slopes.
    3) Square the result.
    4) Apply a short moving average (envelope).
    5) The envelope is used by EngzeeHeartRate for final threshold-based detection.
    """

    def __init__(self, ecg_signal, fs, 
                 lowcut=5.0, highcut=15.0, order=2, 
                 deriv_kernel=None,
                 window_size_sec=0.15):
        
        self.ecg_signal = np.nan_to_num(ecg_signal)
        self.fs = fs
        self.lowcut = max(0.5, lowcut)
        self.highcut = min(0.99 * (fs / 2), highcut)
        self.order = order
        self.deriv_kernel = deriv_kernel
        self.window_size_sec = window_size_sec

        # Outputs
        self.filtered_BandPass = None
        self.derivative_signal = None
        self.envelope_signal   = None  
        self._done_fit         = False

    def bandpass_filter(self):

        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(self.order, [low, high], btype='band')
        self.filtered_BandPass = filtfilt(b, a, self.ecg_signal)

    def derivative_stage(self):

        if self.deriv_kernel is None:

            kernel = np.array([1, 2, 0, -2, -1]) / 8.0
        else:
            kernel = self.deriv_kernel

        self.derivative_signal = np.convolve(self.filtered_BandPass, kernel, mode='same')

    def squaring(self):

        return self.derivative_signal ** 2

    def moving_window_integration(self, squared_signal):

        window_samples = int(self.window_size_sec * self.fs)
        window_samples = max(window_samples, 1)
        window = np.ones(window_samples) / window_samples
        integrated = np.convolve(squared_signal, window, mode='same')
        return integrated

    def fit(self):

        self.bandpass_filter()
        self.derivative_stage()
        squared = self.squaring()
        self.envelope_signal = self.moving_window_integration(squared)

        self._done_fit = True
        return self.envelope_signal


class EngzeeHeartRate:


    def __init__(self, ecg_signal, fs, integrated_signal, bandpassed_signal):

        self.ecg_signal = np.nan_to_num(ecg_signal)
        self.fs = fs
        self.integrated_signal = integrated_signal
        self.bandpassed_signal = bandpassed_signal

    def find_r_peaks(self,
                     threshold_candidates=None,
                     min_rr_interval=0.3,
                     search_window_sec=0.1):
   
        if threshold_candidates is None:
            threshold_candidates = [1.0, 1.5, 2.0, 2.5]

        best_peaks = None
        best_count = 0

        search_window = int(search_window_sec * self.fs)
        min_distance = int(min_rr_interval * self.fs)

        median_val = np.median(self.integrated_signal)
        iqr = (np.percentile(self.integrated_signal, 75)
               - np.percentile(self.integrated_signal, 25))

        for tf in threshold_candidates:
            dynamic_threshold = median_val + tf * iqr

            peaks, _ = find_peaks(self.integrated_signal,
                                  height=dynamic_threshold,
                                  distance=min_distance)

            refined_peaks = []
            for peak in peaks:
                start = max(peak - search_window, 0)
                end   = min(peak + search_window, len(self.bandpassed_signal))
                local_max = start + np.argmax(self.bandpassed_signal[start:end])
               
                refined_peaks.append(local_max)

            refined_peaks = sorted(set(refined_peaks))

            if len(refined_peaks) > best_count:
                best_peaks = refined_peaks
                best_count = len(refined_peaks)

    
        if best_peaks is None:
            best_peaks = []

        return np.array(best_peaks)



def detect_ecg_peaks_engzee(signal, fs):

    engzee = EngzeeTompkins(signal, fs)
    integrated_signal = engzee.fit()
    bandpassed_signal = engzee.filtered_BandPass  
    hr_obj = EngzeeHeartRate(signal, fs, integrated_signal, bandpassed_signal)
    R_peaks = hr_obj.find_r_peaks()
    print(f"Detected {len(R_peaks)} R-peaks with EngzeeTompkins")
    return np.array(R_peaks)

