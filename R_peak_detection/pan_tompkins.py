
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



""" 
old
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class PanTompkinsBasic:
    def __init__(self, ecg_signal, fs,
                 lowcut=1.0, highcut=40.0,
                 order=1):
        self.raw = np.nan_to_num(ecg_signal)
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
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
        # Simple difference (worse than your original kernel)
        self.derivative = np.diff(self.filtered, prepend=0)
        return self.derivative

    def squaring(self):
        self.squared = self.derivative ** 2
        return self.squared

    def moving_window_integration(self, window_size_sec=0.15):
        w = max(int(window_size_sec * self.fs), 1)
        window = np.ones(w) / w
        self.integrated = np.convolve(self.squared, window, mode='same')
        return self.integrated

    def fit(self):
        self.bandpass_filter()
        self.derivative_filter()
        self.squaring()
        return self.moving_window_integration()

class HeartRateBasic:
    def __init__(self, fs, integrated):
        self.fs = fs
        self.int_sig = integrated

    def find_r_peaks(self, threshold_factor=0.5, refractory_sec=0.2):
        threshold = threshold_factor * np.max(self.int_sig)
        distance = int(refractory_sec * self.fs)
        peaks, _ = find_peaks(self.int_sig, height=threshold, distance=distance)
        return peaks

def detect_ecg_peaks(signal, fs):
    pan = PanTompkinsBasic(signal, fs)
    integrated = pan.fit()
    hr = HeartRateBasic(fs, integrated)
    r_peaks = hr.find_r_peaks()
    print(f"Detected {len(r_peaks)} R-peaks (vanilla)")
    return r_peaks """











""" import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class PanTompkins:
    def __init__(self, ecg_signal, fs, lowcut=5.0, highcut=35, order=2):
        self.ecg_signal = np.nan_to_num(ecg_signal)  
        self.fs = fs
        self.lowcut = max(0.5, lowcut)  
        self.highcut = min(0.99 * (fs / 2), highcut)  
        self.order = order

        self.filtered_BandPass = None
        self._derivative = None
        self._squared = None
        self._integrated = None

    def bandpass_filter(self):

        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(self.order, [low, high], btype='band')
        self.filtered_BandPass = filtfilt(b, a, self.ecg_signal)
        return self.filtered_BandPass

    def derivative_filter(self):
        kernel = np.array([1, 2, 0, -2, -1]) / 8.0
        self._derivative = np.convolve(self.filtered_BandPass, kernel, mode='same')
        return self._derivative

    def squaring(self):
        self._squared = self._derivative ** 2
        return self._squared

    def moving_window_integration(self, window_size_sec=0.15):
        window_samples = int(window_size_sec * self.fs)
        window_samples = max(window_samples, 1)
        window = np.ones(window_samples) / window_samples
        self._integrated = np.convolve(self._squared, window, mode='same')
        return self._integrated

    def fit(self):
        self.bandpass_filter()
        self.derivative_filter()
        self.squaring()
        return self.moving_window_integration()



class HeartRate:
    def __init__(self, ecg_signal, fs, integrated_signal, bandpassed_signal):
        self.ecg_signal = np.nan_to_num(ecg_signal)
        self.fs = fs
        self.integrated_signal = integrated_signal
        self.bandpassed_signal = bandpassed_signal

    def find_r_peaks(self,
                     threshold_candidates=None,
                     min_rr_interval=0.3,
                     adaptive_threshold=False,  # New flag for local adaptation
                     local_window_sec=1.0       # Length of local window in seconds
                    ):

        # If no threshold candidates provided, use a default list.
        if threshold_candidates is None:
            threshold_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]

        best_peaks = None
        best_count = 0

        search_window = int(0.1 * self.fs)
        min_distance = int(min_rr_interval * self.fs)

        if not adaptive_threshold:
            # GLOBAL threshold: compute once on the entire integrated_signal.
            median_val = np.median(self.integrated_signal)
            iqr = (np.percentile(self.integrated_signal, 75)
                   - np.percentile(self.integrated_signal, 25))
            # Loop over each candidate scaling factor.
            for tf in threshold_candidates:
                dynamic_threshold = median_val + tf * iqr

                peaks, _ = find_peaks(self.integrated_signal,
                                      height=dynamic_threshold,
                                      distance=min_distance)
                refined_peaks = []
                for peak in peaks:
                    start = max(peak - search_window, 0)
                    end = min(peak + search_window, len(self.bandpassed_signal))
                    local_max = start + np.argmax(self.bandpassed_signal[start:end])
                    refined_peaks.append(local_max)
                refined_peaks = sorted(set(refined_peaks))
                if len(refined_peaks) > best_count:
                    best_peaks = refined_peaks
                    best_count = len(refined_peaks)

        else:
            # ADAPTIVE LOCAL THRESHOLD: Process the integrated signal in segments.
            window_samples = int(local_window_sec * self.fs)
            segments = range(0, len(self.integrated_signal), window_samples)
            all_peaks = []

            for seg_start in segments:
                seg_end = min(seg_start + window_samples, len(self.integrated_signal))
                segment = self.integrated_signal[seg_start:seg_end]
                # Compute local median and IQR.
                local_median = np.median(segment)
                local_iqr = (np.percentile(segment, 75) - np.percentile(segment, 25))
                # Use a fixed candidate or loop over candidates locally.
                for tf in threshold_candidates:
                    local_threshold = local_median + tf * local_iqr
                    seg_peaks, _ = find_peaks(segment,
                                              height=local_threshold,
                                              distance=min_distance)
                    # Adjust indices to match the full signal.
                    seg_peaks = [p + seg_start for p in seg_peaks]
                    all_peaks.extend(seg_peaks)
            # Remove duplicates and sort.
            best_peaks = sorted(set(all_peaks))
            best_count = len(best_peaks)

        return np.array(best_peaks)

def detect_ecg_peaks(signal, fs):
    pan = PanTompkins(signal, fs)
    integrated_signal = pan.fit()
    bandpassed_signal = pan.filtered_BandPass

    hr_obj = HeartRate(signal, fs, integrated_signal, bandpassed_signal)
    R_peaks_fixed = hr_obj.find_r_peaks(threshold_candidates=[1.8168040335044007], min_rr_interval=0.5333302229593873, adaptive_threshold=False)

    R_peaks = hr_obj.find_r_peaks(threshold_candidates=[1.8168040335044007], min_rr_interval=0.3, adaptive_threshold=True, local_window_sec=1.0)


    print(f"Detected {len(R_peaks)} R-peaks with Pan-Tompkins")
    return np.array(R_peaks)   
 """


""" import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class PanTompkins:
    def __init__(self, ecg_signal, fs, lowcut=5.0, highcut=35, order=2):
        self.ecg_signal = np.nan_to_num(ecg_signal)  
        self.fs = fs
        self.lowcut = max(0.5, lowcut)  
        self.highcut = min(0.99 * (fs / 2), highcut)  
        self.order = order

        self.filtered_BandPass = None
        self._derivative = None
        self._squared = None
        self._integrated = None

    def bandpass_filter(self):

        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(self.order, [low, high], btype='band')
        self.filtered_BandPass = filtfilt(b, a, self.ecg_signal)
        return self.filtered_BandPass

    def derivative_filter(self):
        kernel = np.array([1, 2, 0, -2, -1]) / 8.0
        self._derivative = np.convolve(self.filtered_BandPass, kernel, mode='same')
        return self._derivative

    def squaring(self):
        self._squared = self._derivative ** 2
        return self._squared

    def moving_window_integration(self, window_size_sec=0.15):
        window_samples = int(window_size_sec * self.fs)
        window_samples = max(window_samples, 1)
        window = np.ones(window_samples) / window_samples
        self._integrated = np.convolve(self._squared, window, mode='same')
        return self._integrated

    def fit(self):
        self.bandpass_filter()
        self.derivative_filter()
        self.squaring()
        return self.moving_window_integration()



class HeartRate:


    def __init__(self, ecg_signal, fs, integrated_signal, bandpassed_signal):
        self.ecg_signal = np.nan_to_num(ecg_signal)
        self.fs = fs
        self.integrated_signal = integrated_signal
        self.bandpassed_signal = bandpassed_signal

    def find_r_peaks(self,
                     threshold_candidates=None,
                     min_rr_interval=0.3):

        if threshold_candidates is None:
  
            threshold_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]

        best_peaks = None
        best_count = 0

     
        search_window = int(0.1 * self.fs)
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

       
        return np.array(best_peaks)

def detect_ecg_peaks(signal, fs):
    pan = PanTompkins(signal, fs)
    integrated_signal = pan.fit()
    bandpassed_signal = pan.filtered_BandPass

    hr_obj = HeartRate(signal, fs, integrated_signal, bandpassed_signal)
    R_peaks = hr_obj.find_r_peaks()
    print(f"Detected {len(R_peaks)} R-peaks with Pan-Tompkins")
    return np.array(R_peaks) """