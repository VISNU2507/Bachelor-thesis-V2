# Bachelor Thesis: ECG-Derived Respiratory Monitoring

This repository supports the thesis project:

**Transforming Cardiac Signals into Respiratory Insights**  
*Bachelor Thesis in Biomedical Engineering, DTU Health Tech (2025)*  
**Author:** Visnukaran Kirubakaran

---

## 🩺 Project Overview

This project is part of the **DIAMONDS** initiative — a novel system for **non-invasive, continuous respiratory monitoring** using **ECG signals extracted from dEMG recordings**.

The full processing pipeline transforms raw diaphragm EMG signals into respiratory insights via:

- R-peak detection
- EDR (ECG-Derived Respiration) signal computation
- Breath segmentation and classification

---

## 📁 Repository Structure

```
Thesis code - Kopi/
├── Signal_all_code.ipynb               # Core notebook: defining ECG signal, detecting R-peaks, generating EDR signals, plotting, feature table creation

├── preprocessing.py                    # EDR signal creation and filtering logic

├── golden_analysis.py                  # Bland–Altman and MAE/R² comparison against spirometry

├── interpolated_signal_view_2.py       # 🖼Signal interpolation and visualization tool

├── R_peak_detection/                   # R-peak detection and validation utilities
│   ├── peak_detection.ipynb            # Notebook for R-peak detection performance
│   ├── pan_tompkins.py                 # Pan-Tompkins algorithm implementation
│   ├── diamonds_definitions.py         # Shared parameters for detection scripts

├── Breath_Segmentation/                # Breath phase detection + respiratory metric prediction
│   ├── Phase 1/                        
│   │   └── breath_detection.py         # Peak/trough-based adaptive segmentation
│   │   └── phase_1_validation.ipynb    # Validation with Bland–Altman plots
│   └── Phase 2/                         
│       └── phase_2_pipeline.ipynb      # ML prediction pipeline for respiratory features

├── Breath_Classification/              # Final stage: breath-type classification
│   ├── breath_classification_pipeline.ipynb   # XGBoost classification pipeline
│   ├── features_classification.py             # Feature extraction module

├── Datafiles/                          # [Excluded] Due to GDPR restrictions
    ⚠️ Contains subject-level data, feature tables, trained models, and evaluation metrics — **not included in this repository**.
```

---

## 📄 Thesis Reference

This implementation supports the methodology described in:

📘 **`Bachelor_Thesis.pdf`**  
Includes:
- Signal processing pipeline architecture
- Evaluation metrics
---


