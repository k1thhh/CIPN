# 🩺 Chemo Comfort — AI-Based Wearable for Early Detection of CIPN

<p align="center">

**Embedded Healthcare | TinyML | ESP32 | Biomedical Sensing | Edge AI**

</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Biomedical%20Embedded%20Systems-blue" alt="Biomedical Embedded Systems">
  <img src="https://img.shields.io/badge/MCU-ESP32-orange" alt="ESP32">
  <img src="https://img.shields.io/badge/ML-TinyML-green" alt="TinyML">
  <img src="https://img.shields.io/badge/Model-TensorFlow%20Lite-purple" alt="TensorFlow Lite">
  <img src="https://img.shields.io/badge/Application-CIPN-red" alt="CIPN">
</p>

---

## 📌 Overview

**Chemo Comfort** is a portable, low-cost embedded diagnostic prototype designed for the **early, objective screening of Chemotherapy-Induced Peripheral Neuropathy (CIPN)**.

The system combines three complementary assessment methods:

* A simplified **EORTC QLQ-CIPN20 questionnaire**
* **Vibration Perception Threshold (VPT)** testing
* **Cold-temperature perception testing**

The extracted features are processed directly on an **ESP32-WROOM** using a quantized **TensorFlow Lite Micro neural network**, allowing the device to classify CIPN risk without relying on cloud connectivity.

The system produces one of three risk categories:

```text
Normal
Moderate
Severe
```

All processing is performed locally on the device, providing a privacy-preserving architecture suitable for potential home or outpatient applications.

> ⚠️ **Important:** Chemo Comfort is an academic/research prototype and is **not a certified medical device**. It is not intended for clinical diagnosis without further validation.

---

# 🩺 About Chemotherapy-Induced Peripheral Neuropathy

**Chemotherapy-Induced Peripheral Neuropathy (CIPN)** is a common adverse effect associated with several chemotherapy agents, including taxanes, platinum compounds, and vinca alkaloids.

Conventional assessment can involve:

* Patient-reported questionnaires
* Clinical grading scales
* Nerve conduction studies
* Quantitative sensory testing

The project aims to combine subjective patient-reported information with quantitative sensory measurements in a single portable embedded system.

---

# 🎯 Project Objectives

The primary objectives are:

1. Develop a portable embedded system for CIPN screening.
2. Collect patient-reported sensory, motor, and autonomic symptoms.
3. Measure vibration perception thresholds.
4. Measure cold-temperature perception.
5. Combine multiple sensory indicators into a unified feature vector.
6. Perform feature normalization directly on the ESP32.
7. Deploy a quantized TinyML model on-device.
8. Classify CIPN risk into Normal, Moderate, or Severe categories.
9. Display the classification locally using an OLED.
10. Eliminate the requirement for cloud-based inference.
11. Develop a low-cost prototype suitable for further research and validation.

---

# 🏗️ System Architecture

The system combines three input modules before performing edge inference.

```text id="4u9b3c"
┌─────────────────────┐
│   Questionnaire     │
│   OLED + Buttons    │
└──────────┬──────────┘
           │
           │
┌──────────▼──────────┐
│   Vibration Test    │
│    DRV2605L ×2      │
└──────────┬──────────┘
           │
           │
┌──────────▼──────────┐
│   Thermal Test      │
│ Peltier + Temp      │
│      Sensor         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────┐
│       ESP32-WROOM       │
│                         │
│ Feature Normalization   │
│ Mean / Std Scaling      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   TFLite Micro Model    │
│     INT8 Quantized      │
│                         │
│ Dense(16, ReLU)         │
│        ↓                │
│ Dense(1, Sigmoid)       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      OLED Display       │
│ Normal / Moderate /     │
│ Severe                  │
└─────────────────────────┘
```

The five model features are:

```text id="e5q8kj"
VPT
Cold Temperature
CIPN20 Sensory
CIPN20 Motor
CIPN20 Autonomic
```

These features are normalized on-device using the same preprocessing parameters used during model development.

---

# 🔬 Assessment Modules

## 1️⃣ Questionnaire Module

The device presents a simplified **EORTC QLQ-CIPN20-based questionnaire** through the OLED display.

Five push buttons are used for interaction:

* Four questionnaire response buttons
* One reset/response button

The questionnaire contributes sensory, motor, and autonomic symptom features to the final model input.

---

# 2️⃣ Vibration Perception Threshold Test

The VPT test evaluates the user's ability to perceive controlled vibration stimuli.

The system uses **DRV2605L haptic drivers** to generate controlled vibration.

```text id="z3q1n0"
ESP32
  │
  ▼
DRV2605L
  │
  ▼
Increasing Vibration
  │
  ▼
Patient Perception
  │
  ▼
Button Response
  │
  ▼
VPT Threshold
```

The vibration intensity is progressively increased until the patient indicates that the vibration can be perceived.

The project reports VPT threshold trends of approximately:

```text id="kr9n5a"
Non-CIPN → Median 7–8
CIPN     → Median 17–18
```

---

# 3️⃣ Thermal Perception Test

The thermal test evaluates cold-temperature perception.

The system uses:

* TEC1-12706 Peltier module
* DS18B20 / TMP117 temperature sensor
* MOSFET H-bridge

The Peltier module changes the temperature of a contact surface while the temperature sensor continuously measures the resulting temperature.

```text id="n6l4yw"
ESP32
  │
  ▼
MOSFET H-Bridge
  │
  ▼
Peltier Module
  │
  ▼
Cooling Surface
  │
  ▼
Patient Perception
  │
  ▼
Cold Detection Threshold
```

The project reports cold-detection temperatures around:

```text id="d7b5ai"
30.0 – 30.7 °C
```

---

# 🧠 Feature Extraction

The three assessment modules produce the five model inputs:

| Feature            | Source                    |
| ------------------ | ------------------------- |
| `vpt`              | Vibration perception test |
| `cold_temp`        | Thermal perception test   |
| `cipn20_sensory`   | Questionnaire             |
| `cipn20_motor`     | Questionnaire             |
| `cipn20_autonomic` | Questionnaire             |

The complete feature vector is therefore:

```text id="j4bdl7"
[vpt,
 cold_temp,
 cipn20_sensory,
 cipn20_motor,
 cipn20_autonomic]
```

The preprocessing parameters are stored in:

```text id="4px0ck"
ml/model/preproc_params.json
```

and mirrored as constants in the firmware.

---

# 🤖 Machine Learning Pipeline

The machine-learning pipeline was developed using **Python, TensorFlow, and Google Colab**.

```text id="m2h52d"
Clinical / Synthetic Data
          │
          ▼
    Dataset Synthesis
          │
          ▼
     Preprocessing
          │
    ┌─────┴─────┐
    ▼           ▼
Imputation   Standardization
    │           │
    └─────┬─────┘
          ▼
       MLP Model
          │
          ▼
      Training
          │
          ▼
    TensorFlow Lite
          │
          ▼
     INT8 Quantization
          │
          ▼
       model.h
          │
          ▼
       ESP32
```

---

# 📊 Dataset Development

Because a public dataset combining quantitative sensory measurements with CIPN-20 scores was not available for the project's requirements, a **synthetic dataset** was generated.

The dataset was constructed using statistical patterns reported in published literature, with controlled mean and standard-deviation shifts between CIPN and non-CIPN classes.

This approach allowed the embedded ML pipeline to be developed and tested while recognizing that clinical validation with real patient data remains necessary.

---

# ⚙️ Data Preprocessing

The ML pipeline applies two major preprocessing operations.

## Median Imputation

Missing values are handled using:

```text id="u8scnd"
SimpleImputer
      ↓
Median Imputation
```

---

## Feature Standardization

The features are standardized using:

```text id="qzqu2m"
StandardScaler
```

The resulting mean and scaling parameters are saved in:

```text id="5w9f84"
preproc_params.json
```

The same preprocessing parameters are embedded into the ESP32 firmware to maintain consistency between training and inference.

---

# 🧠 Neural Network Architecture

The project uses a compact **two-layer Multilayer Perceptron (MLP)**.

```text id="3f53f5"
5 Input Features
       │
       ▼
┌──────────────────┐
│ Dense Layer      │
│ 16 Neurons       │
│ ReLU Activation  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Dense Layer      │
│ 1 Neuron         │
│ Sigmoid           │
└────────┬─────────┘
         │
         ▼
    CIPN Risk Score
```

The model was trained for:

```text id="4c3vqt"
Epochs      = 30
Optimizer   = Adam
Learning Rate = 0.001
Train/Test  = 85:15
```

---

# 📦 TensorFlow Lite Conversion

After training, the model is converted into an embedded-friendly representation.

```text id="m2trm8"
Keras Model
    ↓
TensorFlow Lite
    ↓
INT8 Post-Training Quantization
    ↓
C Array
    ↓
model.h
    ↓
ESP32 Firmware
```

The resulting `model.h` file contains the TensorFlow Lite model as a C byte array that can be compiled directly into the ESP32 firmware.

---

# ⚡ On-Device TinyML

One of the major features of Chemo Comfort is that inference occurs **directly on the ESP32**.

```text id="d0c4au"
Patient Data
     ↓
Feature Extraction
     ↓
Normalization
     ↓
INT8 TinyML Model
     ↓
Risk Classification
     ↓
OLED
```

No cloud server is required for inference.

This provides:

* Lower dependency on internet connectivity
* Local processing
* Improved privacy
* Potentially faster response
* Portable operation

The firmware executes the model using a **32 KB tensor arena**.

---

# 🖥️ OLED User Interface

A **0.96-inch 128×64 I²C OLED** provides the user interface.

It displays:

* Questionnaire prompts
* Progress
* Test information
* Final CIPN risk classification

The final classification is displayed as:

```text id="5qv8aq"
┌──────────────────┐
│   CIPN RESULT    │
│                  │
│     NORMAL       │
│       OR         │
│    MODERATE      │
│       OR         │
│     SEVERE       │
└──────────────────┘
```

---

# 💻 Firmware Architecture

The main firmware is:

```text id="3myq5u"
firmware/cipn_tinyml.ino
```

It integrates the complete embedded workflow.

The firmware handles:

* OLED interface
* Questionnaire flow
* Button debouncing
* VPT testing
* Thermal testing
* Feature normalization
* TinyML inference
* Risk classification
* Result display

---

# 🔌 Hardware Architecture

| Component              | Function                           |
| ---------------------- | ---------------------------------- |
| **ESP32-WROOM**        | Main controller + TinyML inference |
| **DRV2605L ×2**        | Vibration stimulus generation      |
| **TEC1-12706 Peltier** | Thermal stimulus                   |
| **DS18B20 / TMP117**   | Temperature sensing                |
| **MOSFET H-Bridge**    | Bidirectional Peltier control      |
| **MT3608**             | Boost conversion                   |
| **0.96" OLED**         | User interface                     |
| **5 Push Buttons**     | Questionnaire and test interaction |

The ESP32-WROOM uses a dual-core Xtensa LX6 architecture operating at up to 240 MHz with 520 KB SRAM.

---

# 💰 Cost Analysis

The prototype has an estimated total component cost of approximately:

## **₹1,710**

| Component          | Qty | Unit Cost |        Total |
| ------------------ | --: | --------: | -----------: |
| ESP32-WROOM        |   1 |      ₹400 |         ₹400 |
| DRV2605L           |   2 |      ₹180 |         ₹360 |
| Peltier Module     |   1 |      ₹200 |         ₹200 |
| DS18B20/TMP117     |   1 |      ₹120 |         ₹120 |
| MT3608             |   1 |       ₹80 |          ₹80 |
| OLED               |   1 |      ₹150 |         ₹150 |
| Push Buttons       |   5 |       ₹10 |          ₹50 |
| MOSFET H-Bridge    |   1 |      ₹100 |         ₹100 |
| Power Supply / USB |   1 |      ₹100 |         ₹100 |
| Miscellaneous      |   — |         — |         ₹150 |
| **Total**          |     |           | **≈ ₹1,710** |

---

# 📁 Repository Structure

```text id="y5x4b9"
ChemoComfort-CIPN-Wearable/
│
├── README.md
│
├── docs/
│   └── Chemo_Comfort_Project_Report.pdf
│
├── firmware/
│   ├── cipn_tinyml.ino
│   └── model.h
│
├── ml/
│   ├── notebooks/
│   │   └── cipn.ipynb
│   │
│   ├── data/
│   │   ├── cipn_cancer_drug_only.csv
│   │   └── taxane.csv
│   │
│   └── model/
│       └── preproc_params.json
│
└── PCB/
    ├── esd_pcb_design.kicad_pro
    ├── esd_pcb_design.kicad_sch
    ├── esd_pcb_design.kicad_pcb
    └── esd_pcb_design.kicad_prl
```

The repository therefore combines **embedded firmware, TinyML training artifacts, datasets, preprocessing parameters, and PCB design files** in a single project structure.

---

# 🛠️ Tools & Technologies

## Embedded

* ESP32-WROOM
* Arduino IDE
* Embedded C/C++
* I²C
* GPIO
* PWM
* Sensor interfacing

## Machine Learning

* Python
* TensorFlow
* TensorFlow Lite
* TensorFlow Lite Micro
* Google Colab
* INT8 quantization

## Hardware Design

* KiCad
* PCB schematic design
* PCB layout

## User Interface

* OLED
* Push buttons
* Serial Monitor

---

# 📚 Required Arduino Libraries

The firmware uses:

```text id="c1ytp2"
Adafruit_GFX
Adafruit_SSD1306
Adafruit_DRV2605
tflm_esp32
eloquent_tinyml
Wire
```

`Wire` is included with the Arduino/ESP32 environment.

---

# 🚀 Getting Started

## 1️⃣ Machine Learning Pipeline

Open:

```text id="j9g2xh"
ml/notebooks/cipn.ipynb
```

using Google Colab or Jupyter.

The notebook contains the dataset synthesis, preprocessing, model training, and TensorFlow Lite conversion workflow.

---

## 2️⃣ PCB Design

The PCB design can be opened using **KiCad 8.x or later**.

The main project file is:

```text id="v1p9cq"
PCB/esd_pcb_design.kicad_pro
```

The repository also contains the schematic and PCB layout files.

---

## 3️⃣ Flash the ESP32

Open:

```text id="q1fs93"
firmware/cipn_tinyml.ino
```

Ensure that:

```text id="8l72t4"
model.h
```

is present in the same sketch directory.

Install the required Arduino libraries, select the appropriate ESP32 board and COM port, then upload the firmware.

---

# 🔄 Complete System Workflow

```text id="5a6m2e"
             START
               │
               ▼
        Questionnaire
               │
               ▼
        Sensory Scores
               │
               ▼
        VPT Measurement
               │
               ▼
       Vibration Threshold
               │
               ▼
      Thermal Measurement
               │
               ▼
      Cold Detection Temp
               │
               ▼
      ┌──────────────────┐
      │ 5 Feature Vector │
      └────────┬─────────┘
               │
               ▼
       Median Imputation
               │
               ▼
      Standardized Features
               │
               ▼
       INT8 TinyML Model
               │
               ▼
      CIPN Risk Classification
               │
        ┌──────┼──────┐
        ▼      ▼      ▼
      Normal Moderate Severe
               │
               ▼
             OLED
```

---

# 📊 Project Results

The implemented prototype demonstrates:

| Parameter                     | Result                                |
| ----------------------------- | ------------------------------------- |
| Architecture                  | ESP32-based edge diagnostic prototype |
| Input Modalities              | Questionnaire + VPT + thermal         |
| ML Model                      | 2-layer MLP                           |
| Model Quantization            | INT8                                  |
| Input Features                | 5                                     |
| Tensor Arena                  | 32 KB                                 |
| Inference Time                | **< 1 second**                        |
| Reported AUC                  | **≈ 0.70–0.90**                       |
| PC-to-device parity deviation | **< 2%**                              |
| Estimated Cost                | **≈ ₹1,710**                          |

The reported AUC varies between training runs, while the project reports less than 2% parity deviation from the PC-based Keras model and consistently sub-second ESP32 inference.

---

# 🔍 Verification Strategy

The system can be evaluated at multiple levels.

### Hardware Verification

Verify:

* Vibration generation
* Temperature measurement
* Peltier control
* OLED operation
* Button response

### Sensor Verification

Compare VPT and thermal measurements with expected trends.

### ML Verification

Compare the embedded model's output against the original PC-based Keras model.

### Quantization Verification

Evaluate whether INT8 conversion significantly changes model predictions.

### Embedded Performance

Measure:

* Inference latency
* Memory usage
* Tensor arena requirements
* End-to-end test duration

### System-Level Verification

Verify that questionnaire responses, sensory measurements, preprocessing, inference, and OLED output work as one integrated system.

---

# 🧠 Key Concepts Demonstrated

This project provides practical exposure to:

* Embedded healthcare systems
* Biomedical sensing
* CIPN screening
* ESP32 development
* TinyML
* Edge AI
* TensorFlow Lite Micro
* INT8 quantization
* Neural-network deployment
* Feature normalization
* Median imputation
* Vibration perception testing
* Thermal perception testing
* Peltier control
* Temperature sensing
* Haptic feedback
* OLED interfaces
* Embedded machine learning
* PCB design
* KiCad
* Hardware-software integration

---

# 💡 What I Learned

The project demonstrates how a machine-learning model can move from a Python training environment into a resource-constrained embedded device.

The complete development pipeline can be summarized as:

```text id="qf8qgl"
Data
  ↓
Preprocessing
  ↓
Model Training
  ↓
TensorFlow Lite
  ↓
INT8 Quantization
  ↓
C Array Conversion
  ↓
ESP32 Deployment
  ↓
On-Device Inference
  ↓
Risk Classification
```

At the same time, the project combines **physical sensing, embedded firmware, PCB design, and machine learning** into one end-to-end system.

This makes the project an example of how TinyML can be used to move AI inference closer to the point of data acquisition.

---

# 🏥 Potential Applications

The underlying architecture could potentially be extended to:

### Home Screening

Portable preliminary monitoring outside a hospital environment.

### Outpatient Monitoring

Repeated measurements during chemotherapy treatment.

### Telemedicine

Future wireless connectivity could allow measurements to be shared with healthcare providers.

### Rehabilitation

The same sensory-measurement architecture could potentially be adapted for monitoring changes in peripheral sensory function over time.

> These are potential applications of the prototype; clinical suitability would require substantial real-world validation and regulatory evaluation.

---

# ⚠️ Limitations

The current prototype has several important limitations.

### Synthetic Training Data

The ML dataset was synthesized because a suitable public dataset combining the required sensory measurements and CIPN-20 scores was unavailable.

### Clinical Validation

The model requires validation using real patient data before any clinical interpretation.

### Model Performance

The reported AUC ranges from approximately **0.70–0.90 depending on the training run**, so performance is not yet sufficient to establish clinical reliability.

### Hardware Resolution

The current thermal and vibration measurement mechanisms provide prototype-level sensory measurements rather than certified clinical measurements.

### Regulatory Status

## The system is an academic/research prototype and is **not a certified medical device**.

# 🚀 Future Scope

The project can be extended through:

* Collection of real patient data
* Clinical retraining and validation
* Larger and more diverse datasets
* Electrodermal activity sensing
* Tactile-response timing measurements
* Bluetooth connectivity
* Wi-Fi connectivity
* Telemonitoring
* EHR integration
* Closed-loop PID thermal control
* Higher-resolution temperature threshold measurement
* Touchscreen interface
* Companion mobile application
* Li-ion battery operation
* Portable enclosure
* Medical-device compliance pathway

These extensions are consistent with the project's documented future-work directions.

---

# 📈 Future System Architecture

A future version could evolve toward:

```text id="ynp5i3"
        Multi-Modal Sensing
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
      VPT    Thermal  Questionnaire
       │       │        │
       └───────┼────────┘
               ▼
        Feature Fusion
               │
               ▼
       Edge AI / TinyML
               │
               ▼
       Risk Estimation
               │
       ┌───────┴────────┐
       ▼                ▼
 Local OLED       Wireless Report
                         │
                         ▼
                   Telemonitoring
```

---

# ⭐ Project Highlights

* 🔹 **AI-based CIPN screening prototype**
* 🔹 Multi-modal sensory assessment
* 🔹 EORTC QLQ-CIPN20-based questionnaire
* 🔹 Vibration Perception Threshold testing
* 🔹 Peltier-based thermal perception testing
* 🔹 ESP32-WROOM embedded platform
* 🔹 **On-device TinyML inference**
* 🔹 TensorFlow Lite Micro
* 🔹 **INT8 model quantization**
* 🔹 5-feature ML input vector
* 🔹 2-layer MLP architecture
* 🔹 Sub-second inference
* 🔹 PC-to-device parity within **<2%**
* 🔹 OLED-based user interface
* 🔹 Custom PCB design using KiCad
* 🔹 Estimated prototype cost of **~₹1,710**
* 🔹 Completely offline inference
* 🔹 Privacy-preserving edge architecture

---

# 👥 Authors

**Kirthana S**
**R. Gopikasree**
**Joshitha G**

**Guide:** Dr. V. Prakash

**School of Electronics Engineering**
**Vellore Institute of Technology, Chennai**

The project was developed as part of **BECE403E – Embedded System Design (November 2025)**.

---

# 📌 Keywords

`CIPN` `Chemotherapy-Induced Peripheral Neuropathy` `TinyML` `Edge AI` `ESP32` `TensorFlow Lite` `TensorFlow Lite Micro` `Biomedical Engineering` `Embedded Systems` `Healthcare AI` `Vibration Perception` `Thermal Perception` `Peltier` `DRV2605L` `OLED` `Machine Learning` `INT8 Quantization` `CIPN20` `EORTC` `KiCad` `PCB Design` `Embedded ML` `On-Device Inference`

---

<p align="center">

**Sense → Extract → Normalize → Infer → Classify**

</p>
