import sys
import os


project_root = "C:/Users/Visnu/DIAMONDS"  


if project_root not in sys.path:
    sys.path.append(project_root)


from diamonds import set_data_path, load_patients
import diamonds.data as dt
import diamonds.signal_processing as dsp
import pandas as pd
import diamonds.io.io as dio

data_path = "C:/Users/Visnu/OneDrive - Danmarks Tekniske Universitet/DIAMONDS - Preclinical_new"


set_data_path(data_path)

print("Dataset Path:", data_path)
print("Contents:", os.listdir(data_path)) 

def filtered_patient_list():
    return [
        item for item in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, item)) and not item.lower().endswith(".ini")
    ]

# 🔁 Monkey-patch it into the IO module
dio.get_patient_list = filtered_patient_list

ptt = load_patients(show_progress=True)
pt = ptt[5]


session = pt[dt.SeatedSession]
exercise = session[dt.Exercise]



session_type = dt.SeatedSession  
session = pt[session_type]
        
        
EMG, ECG = pt[session_type, dt.EMG].decompose()
ecg_signal = -ECG.samples[:, 1]  
fs = ECG.fs

import importlib
import preprocessing
importlib.reload(preprocessing)
import pan_tompkins
import engzee_tompkins
import new_rpeak_detector
importlib.reload(pan_tompkins)
importlib.reload(engzee_tompkins)
importlib.reload(new_rpeak_detector)

record = pt

# Then pick the method you want:
method_choice = "pan_tompkins"   # or "engzee" or "pan_tompkins"

# Now call the unified function:
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


__all__ = ['pt', 'session', 'exercise', 'ecg_signal', 'fs', 'QRS_amplitude_resampled']



