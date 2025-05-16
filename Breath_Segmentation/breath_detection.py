import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, iirnotch

DEFAULT_MIN_BREATH_SEC = 3

def generate_overlapping_windows(start_time, end_time, interval_size=10, step_len=5):
    starts = np.arange(start_time, end_time - interval_size, step_len)
    ends = starts + interval_size
    return starts, ends




def detect_edr_breaths(edr_time, edr_signal, min_breath_sec=DEFAULT_MIN_BREATH_SEC):

    min_distance_samples = int(10 * min_breath_sec)
    in_peaks, _ = find_peaks(-edr_signal, distance=min_distance_samples)
    ex_peaks, _ = find_peaks(edr_signal, distance=min_distance_samples)
    in_times = edr_time[in_peaks]
    ex_times = edr_time[ex_peaks]
    return in_times, ex_times

def auto_detect_edr_breaths(edr_time, edr_signal,
                            init_min_breath_sec=3.0,
                            adjust_factor=0.8,
                            max_iterations=5,
                            tol=0.4,
                            min_allowed=0.5,
                            max_allowed=10.0,
                            adaptive=True):

    if not adaptive:
      
        in_times, ex_times = detect_edr_breaths(edr_time, edr_signal, min_breath_sec=init_min_breath_sec)
        print(f"Single-pass detection using min_breath_sec={init_min_breath_sec:.2f}")
        return in_times, ex_times

    current_min_breath_sec = init_min_breath_sec

    for iteration in range(max_iterations):
        in_times, ex_times = detect_edr_breaths(edr_time, edr_signal, min_breath_sec=current_min_breath_sec)
        all_breaths = np.sort(np.concatenate([in_times, ex_times]))
        if len(all_breaths) < 2:
            print(f"Not enough breath events (iteration {iteration+1}). Stopping.")
            return in_times, ex_times
        
        breath_intervals = np.diff(all_breaths)
        median_rough = np.median(breath_intervals)
        upper_cut = 2.0 * median_rough
        lower_cut = 0.2 * median_rough
        cleaned_intervals = breath_intervals[(breath_intervals <= upper_cut) & (breath_intervals >= lower_cut)]
        median_interval = np.median(cleaned_intervals) if len(cleaned_intervals) >= 2 else median_rough

        new_min_breath_sec = adjust_factor * median_interval
        new_min_breath_sec = np.clip(new_min_breath_sec, min_allowed, max_allowed)
        ratio = new_min_breath_sec / current_min_breath_sec
        print(f"Iteration {iteration+1}: old={current_min_breath_sec:.2f}, new={new_min_breath_sec:.2f} (ratio={ratio:.2f})")
        if abs(ratio - 1.0) < tol:
            current_min_breath_sec = new_min_breath_sec
            break
        else:
            current_min_breath_sec = new_min_breath_sec

    in_times, ex_times = detect_edr_breaths(edr_time, edr_signal, min_breath_sec=current_min_breath_sec)
    print(f"Final min_breath_sec after {iteration+1} iteration(s): {current_min_breath_sec:.2f}")
    return in_times, ex_times

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
    plt.xlim(151, 241)
    plt.legend()
    plt.show()




def compute_true_insp_exp_durations(in_times, ex_times, t_start, t_end):
    insp_durations = []
    exp_durations = []

    # ex → In = True Inspiration
    for e_time in ex_times:
        if e_time < t_start or e_time > t_end:
            continue
        next_in = in_times[in_times > e_time]
        if next_in.size > 0 and next_in[0] <= t_end:
            insp_durations.append(next_in[0] - e_time)

    # In → Ex = true Expirationn
    for i_time in in_times:
        if i_time < t_start or i_time > t_end:
            continue
        next_ex = ex_times[ex_times > i_time]
        if next_ex.size > 0 and next_ex[0] <= t_end:
            exp_durations.append(next_ex[0] - i_time)

    return (
        np.mean(insp_durations) if insp_durations else np.nan,
        np.mean(exp_durations) if exp_durations else np.nan
    )

def define_complete_breath_cycles(in_times, ex_times):

    breath_starts = []
    breath_ends = []
    insp_durations = []
    exp_durations = []

    for i in range(len(ex_times) - 1):
        start_ex = ex_times[i]

 
        next_in = in_times[in_times > start_ex]
        if next_in.size == 0:
            continue
        in_time = next_in[0]

   
        next_ex = ex_times[ex_times > in_time]
        if next_ex.size == 0:
            continue
        end_ex = next_ex[0]


        breath_starts.append(start_ex)
        breath_ends.append(end_ex)
        insp_durations.append(in_time - start_ex)
        exp_durations.append(end_ex - in_time)

    return (
        np.array(breath_starts),
        np.array(breath_ends),
        np.array(insp_durations),
        np.array(exp_durations)
    )

def compute_bpm_from_inspiration_times(in_times, t_start, t_end):

    valid = in_times[(in_times >= t_start) & (in_times <= t_end)]
    if valid.size < 2:
        return np.nan
    mean_interval = np.mean(np.diff(valid))
    return 60.0 / mean_interval






def define_breaths_insp_exp(in_times, ex_times):

    breath_starts = []
    breath_ends = []
    
    in_times = np.sort(in_times)
    ex_times = np.sort(ex_times)
    
    for i_time in in_times:
        valid_ex = ex_times[ex_times > i_time]
        if valid_ex.size > 0:
            e_time = valid_ex[0]
            breath_starts.append(i_time)
            breath_ends.append(e_time)
    return np.array(breath_starts), np.array(breath_ends)

def define_breaths_insp_to_expend(edr_time, edr_signal, in_times, ex_times):

    breath_starts = []
    breath_ends = []
    
   
    in_times = np.sort(in_times)
    ex_times = np.sort(ex_times)
    

    for i_time in in_times:
        valid_ex = ex_times[ex_times > i_time]
        if valid_ex.size == 0:
            continue
 
        e_time = valid_ex[0]
  
        next_in = in_times[in_times > e_time]
        if next_in.size > 0:
            window_end = next_in[0]
        else:
            window_end = edr_time[-1]

        mask = (edr_time >= e_time) & (edr_time < window_end)
        if np.any(mask):
            sub_time = edr_time[mask]
            sub_signal = edr_signal[mask]
        
            minima_indices, _ = find_peaks(-sub_signal)
            if minima_indices.size > 0:
                exp_end = sub_time[minima_indices[0]]
            else:
                exp_end = e_time
        else:
            exp_end = e_time
        breath_starts.append(i_time)
        breath_ends.append(exp_end)
    
    return np.array(breath_starts), np.array(breath_ends)