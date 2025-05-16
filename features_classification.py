# features.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.interpolate import interp1d
import sys
import os
from scipy.stats import skew, entropy


project_root = "C:/Users/Visnu/DIAMONDS"  
if project_root not in sys.path:
    sys.path.append(project_root)

import diamonds.data as dt

import importlib
from R_peak_detection import pan_tompkins
importlib.reload(pan_tompkins)
from R_peak_detection.pan_tompkins import detect_ecg_peaks
from R_peak_detection.engzee_tompkins import detect_ecg_peaks_engzee
from R_peak_detection.new_rpeak_detector import detect_r_peaks as neurokit_detect


import importlib
from Breath_Segmentation import breath_detection
importlib.reload(breath_detection)
from Breath_Segmentation.breath_detection import (
    auto_detect_edr_breaths,
    define_breaths_insp_exp,
    define_complete_breath_cycles,
    compute_true_insp_exp_durations,
    DEFAULT_MIN_BREATH_SEC
)
from diamonds_definitions import pt, session, exercise, ecg_signal, fs


record = pt


method_choice = "pan_tompkins"   # or "engzee" 

import importlib
import preprocessing
importlib.reload(preprocessing)
results = preprocessing.extract_ecg_features(
    ecg_signal,
    fs,
    method=method_choice,
    RRI_fs=10,
    default_window_len=0.1,
    search_window_len=0.01,
    edr_method='R_peak'
)

# The dictionary 'results' now has everything:
filtered_ecg = results["filtered_ecg"]
R_peaks       = results["R_peaks"]
RRI           = results["RRI"]
RRI_resampled = results["RRI_resampled"]
QRS_amplitude = results["QRS_amp"]
QRS_amplitude_resampled = results["QRS_amp_resampled"]
BW_effect     = results["BW_effect"]
AM_effect     = results["AM_effect"]
R_peak_amplitude             = results["R_peak_amplitude"]
R_peak_amplitude_resampled   = results["R_peak_amplitude_resampled"]
edr_time_r_peak              = results["edr_time_r_peak"]
edr_signal_r_peak            = results["edr_signal_r_peak"]
edr_rri_time                 = results["edr_rri_time"]
edr_rri_signal               = results["edr_rri_signal"]
combined_time = results["combined_time"] 
combined_signal = results["combined_signal"]

print(f"Method used: {method_choice}")
print(f"Detected {len(R_peaks)} R-peaks")
def _rr_intervals_ms(r_peaks, fs, t0, t1):
    times = r_peaks / fs
    sel = times[(times >= t0) & (times < t1)]
    if len(sel) < 2:
        return np.array([])
    return np.diff(sel) * 1000.0  # ms


def hrv_time_domain(rr_ms):
    if len(rr_ms) < 2:
        return dict(MEANRR=np.nan, SDNN=np.nan, RMSSD=np.nan)
    meanrr = rr_ms.mean()
    sdnn = rr_ms.std(ddof=1)
    diff = np.diff(rr_ms)
    rmssd = np.sqrt((diff**2).mean())

    return dict(MEANRR=meanrr, SDNN=sdnn, RMSSD=rmssd)


# -----------------------------------------------


def heart_rate_from_rpeaks(r_peaks, t0, t1, fs):
    peaks = r_peaks[(r_peaks/fs >= t0) & (r_peaks/fs < t1)]
    if len(peaks) < 2:
        return np.nan
    return 60.0 / np.mean(np.diff(peaks)/fs)


def qrs_stats(edr_t, edr_sig, t0, t1):
    mask = (edr_t >= t0) & (edr_t < t1)
    if not np.any(mask):
        return (np.nan, np.nan)
    seg = edr_sig[mask]
    return seg.mean(), seg.std()


def peak_to_trough(edr_t, edr_sig, t0, t1):
    mask = (edr_t >= t0) & (edr_t < t1)
    if not np.any(mask):
        return np.nan
    seg = edr_sig[mask]
    return seg.max() - seg.min()


def qrs_slope(filtered_ecg, r_peaks, fs, t0, t1, win_ms=10):
    win = int((win_ms/1000)*fs)
    sel = [p for p in r_peaks if t0 <= p/fs < t1 and p-win>=0 and p+win<len(filtered_ecg)]
    if not sel:
        return np.nan
    slopes = [
        np.max(np.abs(np.diff(filtered_ecg[p-win:p+win+1]))) / (1e3/fs)
        for p in sel
    ]
    return np.mean(slopes)




def compute_breath_count(in_times, t_start, t_end):

    valid_in = in_times[(in_times >= t_start) & (in_times <= t_end)]
    return len(valid_in)




import pywt



from scipy.signal import find_peaks

def count_peaks(signal):
    if len(signal) < 10:
        return np.nan
    try:
        peaks, _ = find_peaks(signal)
        return len(peaks)
    except Exception:
        return np.nan



def plot_breath_cycles(edr_time, edr_signal, in_times, ex_times):
    breath_starts, breath_ends = define_breaths_insp_exp(in_times, ex_times)
    
    plt.figure(figsize=(10, 4))
    plt.plot(edr_time, edr_signal, label='EDR Signal')
    
    plt.plot(breath_starts,
             np.interp(breath_starts, edr_time, edr_signal),
             'go', label='Breath Start')
    plt.plot(breath_ends,
             np.interp(breath_ends, edr_time, edr_signal),
             'ro', label='Breath End')

    for t in breath_starts:
        plt.axvline(t, color='green', linestyle='--', alpha=0.5)
    for t in breath_ends:
        plt.axvline(t, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Breath Cycles (Inspiration to Expiration)')
    plt.xlabel('Time (s)')
    plt.ylabel('EDR Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_edr_breaths(edr_time, edr_signal, in_times, ex_times):
    plt.figure(figsize=(10, 4))
    plt.plot(edr_time, edr_signal, label='EDR Signal (QRS Amplitude)')
    plt.plot(in_times,
             np.interp(in_times, edr_time, edr_signal),
             'o', label='Inspirations')
    plt.plot(ex_times,
             np.interp(ex_times, edr_time, edr_signal),
             'o', label='Expirations')
    plt.title('ECG-Derived Respiration with Detected Breaths')
    plt.xlabel('Time (s)')
    plt.ylabel('EDR Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()


def build_feature_table(
    patient, session,
    filtered_ecg, R_peaks, fs,
    QRS_amp_resampled,
    interval_size=10, step_len=5
):

    
    ex_ann = session[dt.Exercise]
    t0 = float(min(ex_ann.timestamp))
    tN = float(max(ex_ann.timestamp)) + 30


    edr_t, edr_sig, _ = QRS_amp_resampled
    in_times, ex_times = auto_detect_edr_breaths(
        edr_t,
        edr_sig,
        init_min_breath_sec=1.2,
        adjust_factor=0.8,
        max_iterations=5,
        tol=0.2,
        min_allowed=0.5,
        max_allowed=10.0,
        adaptive=True
    )

    starts = np.arange(t0, tN - interval_size + 1e-9, step_len)
    rows = []
    for ws in starts:
        we = ws + interval_size
        # breath metrics
        ti, te = compute_true_insp_exp_durations(in_times, ex_times, ws, we)
        ie = ti/te if te>0 and not np.isnan(ti) and not np.isnan(te) else np.nan
        loc_in = in_times[(in_times>=ws)&(in_times<we)]
        b_st, b_en, _, _ = define_complete_breath_cycles(loc_in, ex_times[(ex_times>=ws)&(ex_times<we)])
        n_b = len(b_st)
        avg_tot = (b_en-b_st).mean() if n_b else np.nan
        edr_bpm = 60/np.mean(np.diff(loc_in)) if len(loc_in)>=2 else np.nan

        # ECG/QRS
        hr_ecg = heart_rate_from_rpeaks(R_peaks, ws, we, fs)
        m_qrs, sd_qrs = qrs_stats(edr_t, edr_sig, ws, we)
        p2t = peak_to_trough(edr_t, edr_sig, ws, we)
        slope = qrs_slope(filtered_ecg, R_peaks, fs, ws, we)

        # HRV time-domain + poincaré
        rr_ms = _rr_intervals_ms(R_peaks, fs, ws, we)
        td    = hrv_time_domain(rr_ms)

        # — 1) EDR spectral features via Welch PSD —
        # mask out the segment
        mask_edr = (edr_t >= ws) & (edr_t < we)
        seg = edr_sig[mask_edr]
        if len(seg) >= 8:   # need at least a few samples for PSD
            f, Pxx = welch(seg, fs=10, nperseg=min(len(seg), 256))
            # respiratory band 0.1–0.5 Hz
            band = (f >= 0.1) & (f <= 0.5)
            bandpower    = Pxx[band].sum()
            total_power  = Pxx.sum()
            
            peak_freq     = f[Pxx.argmax()]
            
        else:
            bandpower = total_power = rel_bandpower = peak_freq = centroid = np.nan

        # — 2) simple time‑domain EDR slope/delta —
        # mean of first and last 1 s of the window
        s1 = edr_sig[(edr_t >= ws)     & (edr_t < ws + 1)]
        s2 = edr_sig[(edr_t >= we - 1) & (edr_t < we    )]



        rows.append({
            'subject_id': patient.id,
            'window_start': ws,
            'window_end': we,
            'EDR_BPM': edr_bpm,
            'n_breaths': n_b,
            'AvgTotalBreathDuration': avg_tot,
            'TrueInspDuration': ti,
            'TrueExpDuration': te,
            'IEratio': ie,
            'HR_ECG': hr_ecg,
            'MeanQRSamp': m_qrs,
            'StdQRSamp': sd_qrs,
            'Peak2Trough': p2t,
            'QRSslope': slope,
            **td,
            # new spectral features
            'EDR_band_power':   bandpower,
            'EDR_peak_freq':    peak_freq,


        })

    df = pd.DataFrame(rows).sort_values('window_start').reset_index(drop=True)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].interpolate('linear', limit_direction='both')
    df['n_breaths'] = df['n_breaths'].fillna(0).astype(int)
    return df





def main():
    
    edr_time = QRS_amplitude_resampled[0]
    edr_signal = QRS_amplitude_resampled[2]

    in_times, ex_times = auto_detect_edr_breaths(
        edr_time,
        edr_signal
    )
    
    plot_breath_cycles(edr_time, edr_signal, in_times, ex_times)
    plot_edr_breaths(edr_time, edr_signal, in_times, ex_times)

    session = pt[dt.SeatedSession] 
    exercise_ann = session[dt.Exercise]

    interval_size = 10  
    feature_df = build_feature_table(
        pt, session,
        filtered_ecg,            # pre‑filtered signal
        R_peaks,                 # indices of R peaks
        fs,
        QRS_amplitude_resampled,       # tuple (t, raw, filt)
        interval_size=10, step_len=5
    )
    print("Feature Table:")
    print(feature_df)

   
    labels = [
        f"{int(ws)}–{int(we)}s"
        for ws, we in zip(feature_df["window_start"], feature_df["window_end"])
    ]
   
    mid_times = feature_df['window_start'] + (interval_size / 2)

  
    plt.figure()
    plt.plot(mid_times, feature_df['TrueInspDuration'], 'o-', label='True Insp Duration')
    plt.plot(mid_times, feature_df['TrueExpDuration'], 'o-', label='True Exp Duration')
    plt.title('True Inspiration and Expiration Durations')
    plt.xlabel('Time (s)')
    plt.ylabel('Duration (s)')
    plt.grid(True)
    plt.legend()
    plt.show()

   
    plt.figure()
    plt.plot(mid_times, feature_df['n_breaths'], 'o-', label='Number of Breaths')
    plt.title('Number of Breaths (per 10s window)')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3) Plot: AvgTotalBreathDuration
    plt.figure()
    plt.plot(mid_times, feature_df['AvgTotalBreathDuration'], 'o-', label='AvgTotalBreathDuration')
    plt.title('Average Total Breath Duration (Ex->Ex)')
    plt.xlabel('Time (s)')
    plt.ylabel('Duration (s)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 4) Plot: Breathing Rate per Minute
    plt.figure()
    plt.plot(mid_times, feature_df['EDR_BPM'], 'o-', label='Breathing Rate (BPM)')
    plt.title('Breathing Rate per Minute')
    plt.xlabel('Time (s)')
    plt.ylabel('BPM')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 5) Bar Plot: EDR_BPM
    plt.figure()
    plt.bar(labels, feature_df["EDR_BPM"], color="royalblue", alpha=0.7)
    plt.title("Breathing Rate per Minute (Segmented Intervals)")
    plt.xlabel("Time Intervals")
    plt.ylabel("BPM")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


    plt.figure()
    plt.bar(labels, feature_df["AvgTotalBreathDuration"], color="seagreen", alpha=0.7)
    plt.title("Average Total Breath Duration (Segmented Intervals)")
    plt.xlabel("Time Intervals")
    plt.ylabel("Duration (s)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()