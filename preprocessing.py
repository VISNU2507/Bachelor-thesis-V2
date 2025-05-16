import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
import importlib
from R_peak_detection import pan_tompkins
importlib.reload(pan_tompkins)
from R_peak_detection.pan_tompkins import detect_ecg_peaks as pan_detect
from R_peak_detection.engzee_tompkins import detect_ecg_peaks_engzee
from R_peak_detection.new_rpeak_detector import detect_r_peaks as neurokit_detect

def interpolate_from_R_peaks(loc, val, fs,
                             resampled_fs=10,
                             interp_method="linear",
                             filter_type="band"):
    loc = np.array(loc, dtype=float)
    val = np.array(val, dtype=float)
    if len(loc) < 2:
        return ([], [], [])
    sig_time = loc / fs


    f_interp = interp1d(sig_time, val, kind=interp_method, fill_value="extrapolate")

    new_time = np.arange(sig_time[0], sig_time[-1], 1.0 / resampled_fs)

    if len(new_time) < 2:
        return ([], [], [])
    
    resampled_sig = f_interp(new_time)

    if filter_type == "band":
        b, a = butter(N=2, Wn=[0.1, 0.5], btype='band', fs=resampled_fs)
    elif filter_type == "low":
        b, a = butter(N=2, Wn=0.4, btype='low', fs=resampled_fs)
    else:
        raise ValueError("Invalid filter_type. Choose 'band' or 'low'.")
    
    filtered_sig = filtfilt(b, a, resampled_sig)
    return (new_time, resampled_sig, filtered_sig)

def get_QRS_amplitude(ecg, R_peaks, fs, default_window_len=0.1):

    if len(R_peaks) < 2:
        return np.array([]), (np.array([]), None, np.array([]))

    QRS_amplitude = np.zeros(len(R_peaks))
    RR_samples = np.diff(R_peaks)

    for i in range(len(R_peaks)):
        
        if i == 0 or i == len(R_peaks) - 1:
            half_win = int(default_window_len * fs)
        else:
   
            local_RR = 0.5 * (RR_samples[i-1] + RR_samples[i])
            adaptive_len_sec = min(default_window_len, 0.15 * (local_RR / fs))
            half_win = int(adaptive_len_sec * fs)

        center = R_peaks[i]
        start = max(center - half_win, 0)
        end   = min(center + half_win, len(ecg))
        window = ecg[start:end]

       
        baseline_win = int(0.02 * fs)
        baseline_start = max(center - baseline_win, start)
        baseline_segment = ecg[baseline_start:center]
        baseline_value = np.mean(baseline_segment) if len(baseline_segment) >= 1 else 0.0


        window_corrected = window - baseline_value
        QRS_amplitude[i] = window_corrected.max() - window_corrected.min()

  
    QRS_amplitude_resampled = interpolate_from_R_peaks(
        R_peaks, QRS_amplitude, fs,
        resampled_fs=10,
        interp_method="linear",
        filter_type="band"
    )

    return QRS_amplitude, QRS_amplitude_resampled

def get_RRI_from_signal(signal, fs, RRI_fs=10):
  
    R_peaks = pan_detect(signal, fs)
    RRI = np.diff(R_peaks) / fs  
    

    RRI_resampled = interpolate_from_R_peaks(R_peaks[:-1], RRI, fs, RRI_fs, interp_method="linear", filter_type="band")

    return R_peaks, RRI, RRI_resampled

def get_combined_signal_from_features(QRS_resampled, RRI_resampled,
                                      resampled_fs=10, freq_band=(0.1, 0.5),
                                      weighting="equal", custom_weights=None):

    import numpy as np
    from  scipy.signal import butter, sosfiltfilt

    def _nan_out():
        nan = np.full(2, np.nan)
        return np.array([0., 1.]), nan, nan


    tq, qrs, _ = QRS_resampled
    tr, rri, _ = RRI_resampled
    if tq.size < 2 or tr.size < 2:
        return _nan_out()

    t0, t1 = max(tq[0], tr[0]), min(tq[-1], tr[-1])
    if t1 - t0 < 1 / resampled_fs:
        return _nan_out()

    t = np.arange(t0, t1, 1 / resampled_fs)
    if t.size < 2:
        return _nan_out()

    q = np.interp(t, tq, qrs)
    r = np.interp(t, tr, rri)
    if q.std() == 0 or r.std() == 0:
        return _nan_out()
    q = (q - q.mean()) / q.std()
    r = (r - r.mean()) / r.std()


    if weighting == "equal":
        wq, wr = 0.5, 0.5
    elif weighting == "auto":                     
        vq, vr = q.var(), r.var()
        wq, wr = vr / (vq + vr), vq / (vq + vr)    
    elif weighting == "custom" and custom_weights:
        wq, wr = custom_weights
    else:
        raise ValueError("weighting must be 'equal', 'auto', or 'custom'.")

    edr_raw = wq * q + wr * r

  
    sos = butter(2, freq_band, btype='band', fs=resampled_fs, output='sos')
    edr_filt = sosfiltfilt(sos, edr_raw)

    return t, edr_raw, edr_filt



def detect_r_peaks_with_method(ecg_signal, fs, method="pan_tompkins"):
   
    if method == "pan_tompkins":
        from R_peak_detection.pan_tompkins import detect_ecg_peaks
        return detect_ecg_peaks(ecg_signal, fs)
    elif method == "engzee":
        from R_peak_detection.engzee_tompkins import detect_ecg_peaks_engzee
        return detect_ecg_peaks_engzee(ecg_signal, fs)
    elif method == "neurokit":
        from R_peak_detection.new_rpeak_detector import detect_r_peaks
        return detect_r_peaks(ecg_signal, fs)
    else:
        raise ValueError("method must be 'pan_tompkins', 'engzee', or 'neurokit'.")

def majority_vote_r_peaks(ecg_signal, fs, methods=["pan_tompkins", "engzee", "neurokit"], tolerance=0.05):

    all_peaks = []

    for method in methods:
        try:
            peaks = detect_r_peaks_with_method(ecg_signal, fs, method=method)
            all_peaks.append(np.array(peaks))
        except Exception as e:
            print(f"⚠️ Detector {method} failed: {e}")

 
    combined = np.concatenate(all_peaks)
    combined.sort()


    clustered = []
    i = 0
    while i < len(combined):
        cluster = [combined[i]]
        j = i + 1
        while j < len(combined) and (combined[j] - combined[i]) <= int(tolerance * fs):
            cluster.append(combined[j])
            j += 1

        if len(cluster) >= 2:
            clustered.append(int(np.mean(cluster)))
        i = j

    return np.array(sorted(set(clustered)))



def extract_ecg_features(
    ecg_signal,
    fs,
    method="pan_tompkins",
    RRI_fs=10,
    default_window_len=0.1,
    search_window_len=0.01,
    edr_method='R_peak'
):
    if method == "majority_vote":
        R_peaks = majority_vote_r_peaks(ecg_signal, fs)
    else:
        R_peaks = detect_r_peaks_with_method(ecg_signal, fs, method=method)

    RRI = np.diff(R_peaks) / fs
    RRI_resampled = interpolate_from_R_peaks(R_peaks[:-1], RRI, fs, RRI_fs, "linear", "low")
    QRS_amp, QRS_amp_resampled = get_QRS_amplitude(ecg_signal, R_peaks, fs, default_window_len)
    combined_time, combined_raw_signal, combined_signal = get_combined_signal_from_features(QRS_amp_resampled, RRI_resampled, resampled_fs=10, freq_band=(0.1, 0.5), weighting="equal",
        custom_weights=None)

    return {
        "filtered_ecg": ecg_signal, 
        "R_peaks": R_peaks,
        "RRI": RRI,
        "RRI_resampled": RRI_resampled,
        "QRS_amp": QRS_amp,
        "QRS_amp_resampled": QRS_amp_resampled,
        "combined_time": combined_time,
        "combined_raw_signal": combined_raw_signal,
        "combined_signal": combined_signal
    }


