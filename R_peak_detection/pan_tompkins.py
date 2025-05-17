
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

class PanTompkins:
    def __init__(self, ecg_signal, fs,
                 lowcut=5.0, highcut=15.0,
                 notch_freq=50.0, order=2):
        self.raw = np.nan_to_num(ecg_signal)
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.order = order

        self.filtered = None
        self.derivative = None
        self.squared = None
        self.integrated = None


    def bandpass_filter(self):

     
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        self.filtered = filtfilt(b, a, self.raw)
        return self.filtered

    def derivative_filter(self):
        kernel = np.array([1, 2, 0, -2, -1]) / 8.0
        self.derivative = np.convolve(self.filtered, kernel, mode='same')
        return self.derivative

    def squaring(self):
        self.squared = self.derivative ** 2
        return self.squared

    def moving_window_integration(self, window_size_sec=0.15):
        w = int(window_size_sec * self.fs)
        w = max(w, 1)
        window = np.ones(w) / w
        self.integrated = np.convolve(self.squared, window, mode='same')
        return self.integrated

    def fit(self):
        self.bandpass_filter()
        self.derivative_filter()
        self.squaring()
        return self.moving_window_integration()

class HeartRate:
    def __init__(self, ecg_signal, fs, integrated, bandpassed):
        self.ecg = np.nan_to_num(ecg_signal)
        self.fs = fs
        self.int_sig = integrated
        self.bp_sig = bandpassed

    def find_r_peaks(self,
                    threshold1=None,
                    refractory_sec=0.2):
        if threshold1 is None:
            threshold1 = 0.5 * np.max(self.int_sig)
        threshold2 = 0.5 * threshold1

        SPKI, NPKI = 0.0, 0.0
        peaks, _ = find_peaks(self.int_sig, distance=int(refractory_sec * self.fs))
        signal_peaks = []

        for p in peaks:
            height = self.int_sig[p]

            
            alpha_min, alpha_max = 0.125, 0.3  
            dynamic_range = np.max(self.int_sig) - np.min(self.int_sig) + 1e-6
            norm_height = height / dynamic_range
            w = alpha_min + (alpha_max - alpha_min) * norm_height
            w = np.clip(w, alpha_min, alpha_max)

            if height >= threshold1:
                SPKI = w * height + (1 - w) * SPKI
                signal_peaks.append(p)
            else:
                NPKI = w * height + (1 - w) * NPKI

            threshold1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold2 = 0.5 * threshold1

       
        rr_intervals = np.diff(signal_peaks) / self.fs
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else (60 / self.fs)
        for i in range(len(signal_peaks) - 1):
            if (signal_peaks[i + 1] - signal_peaks[i]) / self.fs > 1.5 * mean_rr:
                seg = self.int_sig[signal_peaks[i]:signal_peaks[i + 1]]
                locs, _ = find_peaks(seg, height=threshold2)
                for loc in locs:
                    signal_peaks.append(signal_peaks[i] + loc)

        signal_peaks = sorted(set(signal_peaks))

       
        refined = []
        half = int(0.1 * self.fs)
        for p in signal_peaks:
            st = max(p - half, 0)
            ed = min(p + half, len(self.bp_sig))
            local = st + np.argmax(self.bp_sig[st:ed])
            refined.append(local)

        return np.array(sorted(set(refined)))



def detect_ecg_peaks(signal, fs, method='pan_tompkins'):
    pan = PanTompkins(signal, fs)
    integrated = pan.fit()
    bp = pan.filtered
    hr = HeartRate(signal, fs, integrated, bp)
    r_peaks = hr.find_r_peaks()
    print(f"Detected {len(r_peaks)} R-peaks")
    return r_peaks













