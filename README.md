# Chemo Comfort — AI-Based Wearable for Early Detection of Chemotherapy-Induced Peripheral Neuropathy (CIPN)

Chemo Comfort is a portable, low-cost embedded diagnostic device for the early, objective screening of **Chemotherapy-Induced Peripheral Neuropathy (CIPN)**. It fuses patient-reported outcomes (a simplified EORTC QLQ-CIPN20 questionnaire), a vibration perception threshold (VPT) test, and a cold-temperature perception test, then runs a TinyML model **on-device** on an ESP32 to classify CIPN risk — with no cloud dependency.

Built as a project for **BECE403E – Embedded System Design**, School of Electronics Engineering, VIT Chennai (Nov 2025).

> Kirthana S (23BEC1412), R. Gopikasree (23BEC1013), Joshitha G (23BEC1478) — Guide: Dr. V. Prakash

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Hardware](#hardware)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Firmware](#firmware)
- [Getting Started](#getting-started)
- [Results](#results)
- [Cost Analysis](#cost-analysis)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

CIPN is a common, debilitating side effect of chemotherapy agents such as taxanes, platinum compounds, and vinca alkaloids. Standard clinical assessment relies on subjective questionnaires (EORTC QLQ-CIPN20) and grading scales (CTCAE), while objective gold-standard tests (NCS, QST) are expensive and clinic-bound.

**Chemo Comfort** closes this gap with a compact device that combines:

1. **Questionnaire module** — a simplified CIPN-20 patient-reported outcome survey on an OLED display, navigated via push buttons.
2. **Vibration perception test (VPT)** — a DRV2605L haptic driver ramps up vibration intensity; the patient presses a button when they feel it, capturing large-fibre sensory threshold.
3. **Thermal perception test** — a Peltier element (TEC1-12706) + DS18B20/TMP117 temperature sensor cools a contact surface to measure small-fibre cold-detection threshold.
4. **Edge ML inference** — an ESP32 runs a quantized TensorFlow Lite Micro model (2-layer MLP: Dense-16 ReLU → Sigmoid) on the 5 extracted features to classify CIPN risk (Normal / Moderate / Severe), displayed on the OLED in under 1 second.

All processing happens on-device — no internet connection required, preserving patient privacy and enabling home/outpatient use.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Questionnaire   │     │  Vibration Test    │     │   Thermal Test      │
│  (OLED + buttons)│     │  (DRV2605L haptic) │     │ (Peltier + DS18B20) │
└────────┬─────────┘     └─────────┬──────────┘     └──────────┬──────────┘
         │                          │                            │
         └──────────────┬───────────┴─────────────┬──────────────┘
                         ▼                         
                 ┌───────────────────┐
                 │  ESP32-WROOM       │
                 │  Feature normalize │
                 │  (mean/std scaler) │
                 └─────────┬──────────┘
                           ▼
                 ┌───────────────────┐
                 │ TFLite Micro model │
                 │  (model.h, INT8)   │
                 └─────────┬──────────┘
                           ▼
                 ┌───────────────────┐
                 │  OLED risk output  │
                 │ Normal/Mod/Severe  │
                 └───────────────────┘
```

**ML features used for inference:** `vpt`, `cold_temp`, `cipn20_sensory`, `cipn20_motor`, `cipn20_autonomic` (see [`ml/model/preproc_params.json`](ml/model/preproc_params.json) for the exact imputation/scaling parameters baked into the firmware).

## Repository Structure

```
ChemoComfort-CIPN-Wearable/
├── README.md
├── docs/
│   └── Chemo_Comfort_Project_Report.pdf     # Full embedded systems project report
├── firmware/
│   ├── cipn_tinyml.ino                      # Arduino/ESP32 sketch (UI, sensors, inference)
│   └── model.h                              # TFLite model compiled to a C byte array
├── ml/
│   ├── notebooks/
│   │   └── cipn.ipynb                       # Data synthesis, training, TFLite conversion
│   ├── data/
│   │   ├── cipn_cancer_drug_only.csv        # Cancer type / drug type / CIPN label dataset
│   │   └── taxane.csv                       # Taxane-specific clinical/sensory dataset
│   └── model/
│       └── preproc_params.json              # Imputer medians + scaler mean/scale (5 features)
└── PCB/
     └──esd_pcb_design.kicad_pro         # KiCad project file
     └── esd_pcb_design.kicad_sch         # Schematic
     └── esd_pcb_design.kicad_pcb         # PCB layout
     └── esd_pcb_design.kicad_prl         # Local project settings
```

> The full embedded systems project report is versioned in `docs/Chemo_Comfort_Project_Report.pdf`.

## Hardware

| Component | Function |
|---|---|
| ESP32-WROOM (dual-core Xtensa LX6, 240 MHz, 520 KB SRAM) | Core processing + on-device ML inference |
| DRV2605L Haptic Driver ×2 | Controlled vibration stimuli for VPT test |
| Peltier module (TEC1-12706) | Controlled heating/cooling for thermal test |
| DS18B20 / TMP117 | Digital temperature sensing |
| MOSFET H-Bridge | Bidirectional current control for Peltier |
| MT3608 boost converter | Stable supply for high-current modules |
| 0.96" OLED (I²C, 128×64) | Questionnaire, progress, and result display |
| 5× push buttons | 4 questionnaire options + 1 reset/response |

Schematic and PCB layout are in [`PCB`](PCB/) — open `esd_pcb_design.kicad_pro` in KiCad (v8.x recommended) to view/edit.

**Estimated BOM cost:** ≈ ₹1,700–1,800 (full breakdown in the project report).

## Machine Learning Pipeline

Developed in Google Colab (Python 3.10, TensorFlow), see [`ml/notebooks/cipn.ipynb`](ml/notebooks/cipn.ipynb):

1. **Dataset synthesis** — since no public dataset combines quantitative sensory data with CIPN-20 scores, a realistic synthetic dataset was generated from statistical patterns in published literature (see [References](#references)), with controlled mean/SD shifts between CIPN and non-CIPN classes.
2. **Preprocessing** — missing values handled via median imputation (`SimpleImputer`); features standardized with `StandardScaler`. Exact parameters are saved in [`ml/model/preproc_params.json`](ml/model/preproc_params.json) and mirrored as constants in the firmware.
3. **Model** — 2-layer MLP: Dense(16, ReLU) → Dense(1, Sigmoid), trained for 30 epochs with Adam (lr = 0.001), 85:15 train/test split.
4. **Conversion** — Keras model → TensorFlow Lite → INT8 post-training quantization → converted to a C header (`model.h`) via a Colab script, ready to compile into the Arduino sketch.

Raw/derived datasets used during exploration and training live in [`ml/data/`](ml/data/).

## Firmware

[`firmware/cipn_tinyml.ino`](firmware/cipn_tinyml.ino) is the full ESP32 sketch. It:

- Drives the OLED (Adafruit_GFX + Adafruit_SSD1306) through the questionnaire flow and debounces the 5 push buttons.
- Controls the DRV2605L haptic driver for the ramped VPT test.
- Runs the thermal test via the Peltier/H-bridge and DS18B20/TMP117.
- Normalizes the 5-feature vector on-device using the same imputer/scaler constants as training (baked in from `preproc_params.json`).
- Loads `model.h` via `tflm_esp32` / `eloquent_tinyml` (`Eloquent::TF::Sequential`) and runs inference in a 32 KB tensor arena.
- Displays the resulting risk category on the OLED.

### Required Arduino libraries

- `Adafruit_GFX`
- `Adafruit_SSD1306`
- `Adafruit_DRV2605`
- `tflm_esp32`
- `eloquent_tinyml`
- `Wire` (bundled)

### Flashing

1. Open `firmware/cipn_tinyml.ino` in Arduino IDE with `model.h` in the same sketch folder.
2. Install the libraries above via Library Manager.
3. Select an ESP32-WROOM-DEV board, set the correct COM port.
4. Upload, then open Serial Monitor (baud matching the sketch) to view VPT/threshold/inference logs.

## Getting Started

```bash
git clone https://github.com/<your-username>/ChemoComfort-CIPN-Wearable.git
cd ChemoComfort-CIPN-Wearable

# Retrain / inspect the ML pipeline
cd ml/notebooks
jupyter notebook cipn.ipynb

# Open hardware design
# (requires KiCad 8+)
open ../../hardware/kicad/esd_pcb_design.kicad_pro

# Flash firmware
# open firmware/cipn_tinyml.ino in Arduino IDE
```

## Results

- Vibration test thresholds matched literature trends: median 7–8 (non-CIPN) vs 17–18 (CIPN).
- Thermal test correctly recorded cold-detection temperatures around 30.0–30.7 °C.
- On-device TinyML model: AUC ≈ 0.70–0.90 depending on training run (see `docs/` report, Fig. 3.1/3.2), with < 2% parity deviation from the PC-based Keras model.
- Inference latency: consistently < 1 second on ESP32.

Full methodology, evaluation plots, and discussion are in the project report (`docs/Chemo_Comfort_Project_Report.pdf`, add it to `docs/`).

## Cost Analysis

| Component | Qty | Unit Cost (₹) | Total (₹) |
|---|---|---|---|
| ESP32-WROOM Module | 1 | 400 | 400 |
| DRV2605L Haptic Driver | 2 | 180 | 360 |
| Peltier Module (TEC1-12706) | 1 | 200 | 200 |
| DS18B20/TMP117 Temp Sensor | 1 | 120 | 120 |
| MT3608 Boost Converter | 1 | 80 | 80 |
| OLED Display (0.96") | 1 | 150 | 150 |
| Push Buttons | 5 | 10 | 50 |
| MOSFET H-Bridge Module | 1 | 100 | 100 |
| Power Supply / USB Cable | 1 | 100 | 100 |
| Misc. (wires, PCB, connectors) | — | — | 150 |
| **Total** | | | **≈ ₹1,710** |

## Future Work

- Collect real patient data from oncology centers for clinical-grade retraining/validation.
- Add electrodermal activity / tactile response timing sensors for multi-fibre analysis.
- Bluetooth/Wi-Fi connectivity for telemonitoring and EHR integration.
- Closed-loop PID thermal control for higher-resolution thresholds.
- Touchscreen or companion mobile app for accessibility.
- Li-ion battery-powered portable enclosure.
- Pursue IEC/ISO medical device compliance pathway.

## References

1. EORTC QLQ-CIPN20 Chemotherapy-Induced Peripheral Neuropathy Questionnaire (English Version), 2018.
2. Kim S. "Predicting Chemotherapy-Induced Peripheral Neuropathy Using Transformer-Based Multimodal Deep Learning." *Research* (Science Partner Journal), 2025;8:Article 0795.
3. Full reference list (20+ literature sources used for synthetic dataset construction and clinical benchmarking) is in the project report.

This is an academic/research prototype and **is not a certified medical device**; it is not intended for clinical diagnosis without further validation.
