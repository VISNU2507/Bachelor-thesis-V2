import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

def interpolate_heart_rate(time_inst_hr, heart_rate, resampled_fs=10):
    hr_interpolator = interp1d(time_inst_hr, heart_rate, kind='linear', fill_value='extrapolate')
    t_start = time_inst_hr[0]
    t_end   = time_inst_hr[-1]
    new_time = np.arange(t_start, t_end, 1 / resampled_fs)
    raw_interpolated = hr_interpolator(new_time)
    b, a = butter(2, 0.5, btype='low', fs=resampled_fs)
    filtered_hr = filtfilt(b, a, raw_interpolated)
    return new_time, raw_interpolated, filtered_hr

def plot_interpolated_signals(
    ecg_signal, filtered_ecg, R_peaks,
    RRI, RRI_resampled,
    QRS_amplitude, QRS_amplitude_resampled,
    fs,
    combined_time, combined_signal, combined_raw_signal,
    R_peaks_engzee=None, R_peak_new_2=None,
):

    fig, axe = plt.subplots(5, 1, figsize=(20, 28))
    axe = list(axe) 


    t2 = np.arange(0, len(ecg_signal)) / fs


    axe[0].plot(t2, filtered_ecg, label='Filtered ECG')
    axe[0].plot(t2[R_peaks], filtered_ecg[R_peaks], 'rx', label='R-peaks (PanTompkins)')
    axe[0].set_title('Filtered ECG with R-peaks')
    axe[0].set_xlabel('Time (s)')
    axe[0].set_ylabel('Amplitude (mV)')
    axe[0].set_xlim(12, 352)
    axe[0].set_ylim(-800, 4000)
    axe[0].legend()
    
    if R_peaks_engzee is not None and len(R_peaks_engzee) > 0:
        axe[0].plot(t2[R_peaks_engzee], filtered_ecg[R_peaks_engzee], 'go',
                    label='R-peaks (Engzee)', alpha=0.7)
        axe[0].legend()
    if R_peak_new_2 is not None and len(R_peak_new_2) > 0:
        axe[0].plot(t2[R_peak_new_2], filtered_ecg[R_peak_new_2], 'y*',
                    label='R-peaks (New 2)', alpha=0.9)
        axe[0].legend()

    

    axe[1].plot(t2[R_peaks[:-1]], RRI, 'o-', label='Original RRI', color='tab:blue')
    axe[1].plot(RRI_resampled[0], RRI_resampled[1], label='Interpolated RRI', color='deepskyblue')
    axe[1].set_title('Interpolated RRI')
    axe[1].set_xlabel('Time (s)')
    axe[1].set_ylabel('Interval (s)')
    axe[1].set_xlim(100, 300)
    axe[1].set_ylim(0.5, 1.2)
    axe[1].legend()
    

    axe[2].plot(t2[R_peaks], QRS_amplitude, 's-', label='Original QRS Amp', color='tab:red')
    axe[2].plot(QRS_amplitude_resampled[0], QRS_amplitude_resampled[1], label='Interpolated QRS Amp', color='lightcoral')
    axe[2].set_title('QRS Amplitude Over Time')
    axe[2].set_xlabel('Time (s)')
    axe[2].set_ylabel('Amplitude (mV)')
    axe[2].set_xlim(100, 300)
    axe[2].set_ylim(190, 1000)
    axe[2].legend()
    

    heart_rate = 60.0 / RRI
    time_inst_hr = (t2[R_peaks[:-1]] + t2[R_peaks[1:]]) / 2.0
    new_time_hr, hr_linear, hr_filtered = interpolate_heart_rate(time_inst_hr, heart_rate, resampled_fs=10)
    axe[3].plot(time_inst_hr, heart_rate, 'ro', label="Original HR Points")
    axe[3].plot(new_time_hr, hr_linear, 'g--', label="Linear Interp HR")
    axe[3].plot(new_time_hr, hr_filtered, 'b-', label="Interpolated + Filtered HR")
    axe[3].set_title("Interpolated Heart Rate Over Time")
    axe[3].set_xlabel("Time (s)")
    axe[3].set_ylabel("Heart Rate (BPM)")
    axe[3].set_xlim(151, 181)
    axe[3].set_ylim(60, 120)
    axe[3].legend()


    axe[4].plot(combined_time, combined_raw_signal, label='Combined EDR (QRS + RRI)', color='orange')
    axe[4].set_title('Combined EDR Signal (Normalized QRS + RRI)')
    axe[4].set_xlabel('Time (s)')
    axe[4].set_ylabel('Amplitude (a.u.)')
    axe[4].legend()
    axe[4].grid(True)


    if len(combined_time) > 0:
        axe[4].set_xlim(combined_time[0], combined_time[-1])
    else:
        axe[4].set_xlim(0, 1) 



    
    plt.tight_layout()
    plt.show()
