# features.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
import sys
import os

project_root = "C:/Users/Visnu/DIAMONDS"  
if project_root not in sys.path:
    sys.path.append(project_root)

import diamonds.data as dt
import importlib
from R_peak_detection import pan_tompkins
importlib.reload(pan_tompkins)
from R_peak_detection.pan_tompkins import detect_ecg_peaks





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

import importlib
import preprocessing
importlib.reload(preprocessing)

record = pt


method_choice = "pan_tompkins"  

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


filtered_ecg = ecg_signal
R_peaks       = results["R_peaks"]
RRI           = results["RRI"]
RRI_resampled = results["RRI_resampled"]
QRS_amplitude = results["QRS_amp"]
QRS_amplitude_resampled = results["QRS_amp_resampled"]
combined_time = results["combined_time"] 
combined_signal = results["combined_signal"]

print(f"Method used: {method_choice}")
print(f"Detected {len(R_peaks)} R-peaks")




def compute_breath_count(in_times, t_start, t_end):

    valid_in = in_times[(in_times >= t_start) & (in_times <= t_end)]
    return len(valid_in)



def compute_peak_to_trough(edr_time, edr_signal, t_start, t_end):
    mask = (edr_time >= t_start) & (edr_time < t_end)
    segment = edr_signal[mask]
    if segment.size == 0:
        return np.nan
    return np.max(segment) - np.min(segment)


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
    plt.xlim(100, 300)
    plt.legend()
    plt.show()


def build_feature_table(
    patient,
    session,
    ecg_signal,
    fs,
    QRS_amplitude_resampled,
    interval_size: float = 10,
    step_len: float = 5,
):



    exercises         = session[dt.Exercise]
    all_ex_timestamps = np.array(exercises.timestamp, dtype=float)
    full_start        = float(all_ex_timestamps.min())
    full_end = float(all_ex_timestamps.max()) + 30  

    print(f"build_feature_table: Using exercise interval {full_start:.1f}–{full_end:.1f} s")


    edr_time_qrs   = QRS_amplitude_resampled[0]
    edr_signal_qrs = QRS_amplitude_resampled[2]


    in_times, ex_times = auto_detect_edr_breaths(
        edr_time_qrs,
        edr_signal_qrs,
        init_min_breath_sec=1.2,
        adjust_factor=0.8,
        max_iterations=5,
        tol=0.2,
        min_allowed=0.5,
        max_allowed=10.0,
        adaptive=True
    )


    seg_starts = np.arange(
        full_start,
        full_end - interval_size + 1e-9,
        step_len
    )

    feature_rows = []
    for window_start in seg_starts:
        window_end = window_start + interval_size

       
        true_insp, true_exp = compute_true_insp_exp_durations(
            in_times, ex_times, window_start, window_end
        )
        ie_ratio = (
            true_insp / true_exp
            if not (np.isnan(true_insp) or np.isnan(true_exp)) and true_exp > 0
            else np.nan
        )

       
        local_ex = ex_times[(ex_times >= window_start) & (ex_times < window_end)]
        local_in = in_times[(in_times >= window_start) & (in_times < window_end)]
        b_starts, b_ends, _, _ = define_complete_breath_cycles(local_in, local_ex)
        n_breaths = len(b_starts)

        if n_breaths > 0:
            total_durations = b_ends - b_starts
            avg_total_breath_duration = np.mean(total_durations)
        else:
            avg_total_breath_duration = np.nan

 
        valid_ins = in_times[(in_times >= window_start) & (in_times < window_end)]
        if valid_ins.size >= 2:
            edr_bpm = 60.0 / np.mean(np.diff(valid_ins))
        else:
            edr_bpm = np.nan

        feature_rows.append({
            "subject_id":             patient.id,
            "window_start":           window_start,
            "window_end":             window_end,
            "EDR_BPM":                edr_bpm,
            "n_breaths":              n_breaths,
            "AvgTotalBreathDuration": avg_total_breath_duration,
            "TrueInspDuration":       true_insp,
            "TrueExpDuration":        true_exp,
            "IEratio":                ie_ratio
        })


    feature_df = pd.DataFrame(feature_rows)
    feature_df = feature_df.sort_values("window_start").reset_index(drop=True)


    num_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[num_cols] = (
        feature_df[num_cols]
        .interpolate(method="linear", limit_direction="both", axis=0)
    )


    feature_df["n_breaths"] = feature_df["n_breaths"].fillna(0).astype(int)

    return feature_df





def main():
    
    edr_time = QRS_amplitude_resampled[0]
    edr_signal = QRS_amplitude_resampled[2]

    in_times, ex_times = auto_detect_edr_breaths(
        edr_time,
        edr_signal,
        init_min_breath_sec=1.2,
        adjust_factor=0.8,
        max_iterations=5,
        tol=0.2,
        min_allowed=0.5,
        max_allowed=10.0,
        adaptive=True
    )
    
    plot_breath_cycles(edr_time, edr_signal, in_times, ex_times)
    plot_edr_breaths(edr_time, edr_signal, in_times, ex_times)

    session = pt[dt.SeatedSession] 
    exercise_ann = session[dt.Exercise]

    interval_size = 10  
    feature_df = build_feature_table(
        record, session, ecg_signal, fs,
        QRS_amplitude_resampled, interval_size=interval_size, step_len=5
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


    plt.figure()
    plt.plot(mid_times, feature_df['AvgTotalBreathDuration'], 'o-', label='AvgTotalBreathDuration')
    plt.title('Average Total Breath Duration (Ex->Ex)')
    plt.xlabel('Time (s)')
    plt.ylabel('Duration (s)')
    plt.grid(True)
    plt.legend()
    plt.show()

  
    plt.figure()
    plt.plot(mid_times, feature_df['EDR_BPM'], 'o-', label='Breathing Rate (BPM)')
    plt.title('Breathing Rate per Minute')
    plt.xlabel('Time (s)')
    plt.ylabel('BPM')
    plt.grid(True)
    plt.legend()
    plt.show()

 
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

